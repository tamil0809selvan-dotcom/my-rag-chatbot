import os
import io
import time
import base64
import hashlib
import streamlit as st
import faiss
import numpy as np

from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors, types
from streamlit_mic_recorder import mic_recorder


# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"

st.set_page_config(
    page_title="PDF Chat | RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0e1117;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid #303644;
        border-radius: 18px;
        background: linear-gradient(135deg, #1c2230, #11141b);
        margin-bottom: 1rem;
    }

    .muted {
        color: #a6adbb;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed #50596a;
        border-radius: 14px;
        padding: .5rem;
    }

    div.stButton > button {
        border-radius: 11px;
        min-height: 2.5rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #292e39;
        border-radius: 14px;
        padding: .8rem;
    }

    div[data-testid="stAudio"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULTS = {
    "active_mode": None,
    "messages": [],
    "pdf_hash": None,
    "pdf_name": None,
    "pdf_ready": False,
    "chunks": [],
    "faiss_index": None,
    "embedding_model": None,
    "last_voice_hash": None,
    "last_auto_pdf_hash": None,
    "last_transcript": None,
    "tts_audio": {},
    "tts_errors": {},
    "autoplay_audio_id": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

client = genai.Client(api_key=API_KEY) if API_KEY else None


# =========================================================
# 3. EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# 4. GEMINI TEXT RESPONSE WITH RETRY
# =========================================================

def ask_gemini_with_retry(prompt, max_retries=5):

    if client is None:
        raise ValueError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    retryable_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except errors.APIError as error:
            if (
                error.code not in retryable_codes
                or attempt == max_retries - 1
            ):
                raise

            wait_seconds = min(2 ** (attempt + 1), 20)

            st.warning(
                f"Gemini temporarily busy ({error.code}). "
                f"Retrying in {wait_seconds} seconds... "
                f"({attempt + 1}/{max_retries - 1})"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Gemini request failed after retries.")


# =========================================================
# 5. VOICE INPUT: SPEECH TO TEXT WITH RETRY
# =========================================================

def transcribe_audio(audio_bytes, mime_type="audio/wav"):

    if client is None:
        raise ValueError(
            "Check your GEMINI_API_KEY in .env."
        )

    retryable_codes = {429, 500, 502, 503, 504}

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    (
                        "Transcribe the user's speech accurately. "
                        "The speech may be in Tamil, Tanglish "
                        "(Tamil spoken using English letters), "
                        "or English. Preserve the original language "
                        "and wording. Do not translate, answer the "
                        "question, or add explanations. Return only "
                        "the transcribed words."
                    ),
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                ],
            )

            transcript = (response.text or "").strip()

            if not transcript:
                raise ValueError(
                    "Speech could not be converted to text. "
                    "Please record again."
                )

            return transcript

        except errors.APIError as error:
            if (
                error.code not in retryable_codes
                or attempt == 4
            ):
                raise

            wait_seconds = min(2 ** (attempt + 1), 20)

            st.warning(
                f"Voice service temporarily busy ({error.code}). "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        "Voice transcription failed after retries."
    )


# =========================================================
# 6. VOICE OUTPUT: TEXT TO SPEECH
# =========================================================

@st.cache_data(show_spinner=False, max_entries=100)
def text_to_speech(text):
    """Generate and cache audio for each unique answer."""

    if client is None:
        raise ValueError(
            "Check your GEMINI_API_KEY in .env."
        )

    interaction = client.interactions.create(
        model=TTS_MODEL,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text[:4000],
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": (
                                    "Speak naturally and clearly, "
                                    "using the language of the text. "
                                    "Do not translate it."
                                ),
                            }
                        ],
                    }
                ],
            }
        ],
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [{"voice": "Kore"}],
        },
    )

    output_audio = getattr(
        interaction, "output_audio", None
    )

    audio_data = (
        getattr(output_audio, "data", None)
        if output_audio
        else None
    )

    if not audio_data:
        raise ValueError("Gemini did not return audio.")

    if isinstance(audio_data, str):
        audio_bytes = base64.b64decode(audio_data)
    else:
        audio_bytes = bytes(audio_data)

    if not audio_bytes:
        raise ValueError("Gemini returned empty audio.")

    return audio_bytes


# =========================================================
# 7. PDF PROCESSING
# =========================================================

def process_pdf(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text() or ""

        if page_text.strip():
            text_parts.append(
                f"[Page {page_number}]\n{page_text}"
            )

    pdf_text = "\n\n".join(text_parts).strip()

    if not pdf_text:
        raise ValueError(
            "No readable text found. "
            "Scanned PDFs may require OCR."
        )

    chunk_size = 1000
    overlap = 150
    step = chunk_size - overlap

    chunks = [
        pdf_text[start:start + chunk_size].strip()
        for start in range(0, len(pdf_text), step)
    ]

    chunks = [chunk for chunk in chunks if chunk]

    if not chunks:
        raise ValueError("No text chunks were created.")

    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, model


# =========================================================
# 8. PDF RETRIEVAL
# =========================================================

def retrieve_context(question, top_k=5):

    index = st.session_state.faiss_index
    chunks = st.session_state.chunks
    model = st.session_state.embedding_model

    if index is None or not chunks or model is None:
        return ""

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    context_parts = []

    for chunk_index in indices[0]:
        if chunk_index >= 0:
            context_parts.append(chunks[chunk_index])

    return "\n\n".join(context_parts)


# =========================================================
# 9. CHAT HISTORY
# =========================================================

def build_history():

    history = st.session_state.messages[:-1][-10:]
    lines = []

    for message in history:
        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        lines.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(lines)


# =========================================================
# 10. PROMPT CREATION
# =========================================================

def build_prompt(question, context=""):

    history = build_history()

    if st.session_state.active_mode == "PDF":
        return f"""
You are a helpful PDF question-answering assistant.

Rules:
1. Answer using the retrieved PDF context.
2. Explain clearly and simply.
3. Reply in the same language or style as the question.
4. Tanglish questions can receive Tanglish answers.
5. If the context does not contain the answer, say:
   "Indha PDF-la indha question-ku answer kidaikkala."
6. Do not invent information or claim unsupported facts.

Conversation history:
{history}

Retrieved PDF context:
{context}

Current question:
{question}

Give the answer as normal readable text.
"""

    return f"""
You are a helpful general AI assistant.

Rules:
1. Answer the user's question directly.
2. Use general knowledge where appropriate.
3. Explain clearly using simple language.
4. Reply in the same language or style as the question.
5. Use conversation history for follow-up questions.
6. Do not show internal reasoning or debug information.

Conversation history:
{history}

Current question:
{question}

Give the answer as normal readable text.
"""


# =========================================================
# 11. NAVIGATION
# =========================================================

def go_home():

    st.session_state.active_mode = None
    st.rerun()


def start_new_chat():

    st.session_state.messages = []
    st.session_state.last_voice_hash = None
    st.session_state.last_transcript = None
    st.session_state.tts_audio = {}
    st.session_state.tts_errors = {}
    st.session_state.autoplay_audio_id = None
    st.rerun()


# =========================================================
# 12. SHARED QUESTION HANDLER
# =========================================================

def submit_question(question):

    question = (question or "").strip()

    if not question:
        return False

    if client is None:
        st.error(
            "GEMINI_API_KEY-ai .env file-la check pannu."
        )
        return False

    if (
        st.session_state.active_mode == "PDF"
        and not st.session_state.pdf_ready
    ):
        st.warning(
            "First PDF upload panni process pannu."
        )
        return False

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner("Answer prepare pannudhu..."):

            if st.session_state.active_mode == "PDF":
                context = retrieve_context(question)

                if not context.strip():
                    answer = (
                        "PDF-la readable text retrieve panna mudiyala. "
                        "PDF-ai check panni try pannu."
                    )
                else:
                    prompt = build_prompt(question, context)
                    answer = ask_gemini_with_retry(prompt)

            else:
                prompt = build_prompt(question)
                answer = ask_gemini_with_retry(prompt)

        assistant_message = {
            "role": "assistant",
            "content": answer,
        }

        st.session_state.messages.append(
            assistant_message
        )

        answer_id = hashlib.sha256(
            answer.encode("utf-8")
        ).hexdigest()[:20]

        if answer_id not in st.session_state.tts_audio:
            try:
                with st.spinner(
                    "Answer ready! Voice-um prepare pannudhu..."
                ):
                    audio_bytes = text_to_speech(answer)

                st.session_state.tts_audio[answer_id] = audio_bytes
                st.session_state.autoplay_audio_id = answer_id

                st.session_state.tts_errors.pop(
                    answer_id,
                    None,
                )

            except Exception as voice_error:
                st.session_state.tts_errors[answer_id] = (
                    str(voice_error)
                )

        return True

    except errors.APIError as error:

        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            st.session_state.messages.pop()

        if error.code == 429:
            st.error(
                "Gemini API quota exceeded. "
                "Check your API quota and try again later."
            )

        elif error.code in {401, 403}:
            st.error(
                "Gemini API authentication failed. "
                "Check your API key and permissions."
            )

        elif error.code == 503:
            st.error(
                "Gemini server is still busy, bro. "
                "Please wait a little and try again."
            )

        else:
            st.error("Gemini API request failed.")
            st.caption(str(error))

        return False

    except Exception as error:

        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
        ):
            st.session_state.messages.pop()

        st.error("Chat request fail aayiduchu.")
        st.caption(str(error))

        return False


# =========================================================
# 13. HOME PAGE
# =========================================================

if st.session_state.active_mode is None:

    st.title("🤖 PDF Chat")

    st.caption(
        "Ask questions, understand documents, and chat naturally."
    )

    st.divider()

    st.write(
        "Vanakkam bro! Welcome to PDF Chat — "
        "unga questions-ai kelunga, naan help panren!"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 PDF Mode")

        st.write(
            "Upload a PDF and ask questions about its content."
        )

        if st.button(
            "Open PDF Mode",
            use_container_width=True,
        ):
            st.session_state.active_mode = "PDF"
            st.session_state.messages = []
            st.rerun()

    with col2:
        st.subheader("🧠 General AI Mode")

        st.write(
            "Ask general questions without uploading a PDF."
        )

        if st.button(
            "Open General AI",
            use_container_width=True,
        ):
            st.session_state.active_mode = "GENERAL"
            st.session_state.messages = []
            st.rerun()

    st.caption(
        "Tamil, Tanglish and English supported."
    )

    st.stop()


# =========================================================
# 14. PDF UPLOAD — AUTOMATIC PROCESSING
# =========================================================

if st.session_state.active_mode == "PDF":

    if not st.session_state.pdf_ready:

        st.markdown("## 📄 Upload your PDF")

        st.caption(
            "Select a document. Processing starts automatically. "
            "No separate Process button is required."
        )

        st.divider()

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pdf_upload_auto",
        )

        if uploaded_file is not None:

            pdf_bytes = uploaded_file.getvalue()

            current_hash = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            if (
                st.session_state.last_auto_pdf_hash
                != current_hash
            ):
                try:
                    with st.spinner(
                        "PDF process pannudhu... Please wait."
                    ):

                        if (
                            current_hash == st.session_state.pdf_hash
                            and st.session_state.faiss_index is not None
                        ):
                            st.session_state.pdf_name = (
                                uploaded_file.name
                            )

                        else:
                            chunks, index, model = process_pdf(
                                pdf_bytes
                            )

                            st.session_state.chunks = chunks
                            st.session_state.faiss_index = index
                            st.session_state.embedding_model = model
                            st.session_state.pdf_hash = current_hash
                            st.session_state.pdf_name = (
                                uploaded_file.name
                            )

                        st.session_state.messages = []
                        st.session_state.pdf_ready = True
                        st.session_state.last_auto_pdf_hash = (
                            current_hash
                        )

                    st.success("PDF processed successfully!")
                    st.rerun()

                except Exception as error:
                    st.error("PDF process panna mudiyala.")
                    st.caption(str(error))

        if st.button("⬅️ Back to Home"):
            go_home()

        st.stop()


# =========================================================
# 15. CHAT HEADER
# =========================================================

if st.session_state.active_mode == "PDF":

    st.markdown("## 📄 PDF Chat")

    st.caption(
        f"Document: {st.session_state.pdf_name or 'Ready'}"
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Home", use_container_width=True):
            go_home()

    with col2:
        if st.button(
            "🔄 Change PDF",
            use_container_width=True,
        ):
            st.session_state.pdf_ready = False
            st.session_state.messages = []
            st.session_state.last_auto_pdf_hash = None
            st.session_state.pdf_hash = None
            st.session_state.pdf_name = None
            st.session_state.chunks = []
            st.session_state.faiss_index = None
            st.session_state.embedding_model = None
            st.rerun()

    with col3:
        if st.button(
            "🗑️ New Chat",
            use_container_width=True,
        ):
            start_new_chat()

else:

    st.markdown("## 🧠 General AI Chat")

    st.caption(
        "Ask questions in Tamil, Tanglish or English."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Home", use_container_width=True):
            go_home()

    with col2:
        if st.button(
            "🗑️ New Chat",
            use_container_width=True,
        ):
            start_new_chat()

st.caption(
    "Answers appear as text. Voice is prepared automatically "
    "when possible. Audio playback may require pressing Play."
)


# =========================================================
# 16. DISPLAY CHAT HISTORY + AUDIO
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            answer_id = hashlib.sha256(
                message["content"].encode("utf-8")
            ).hexdigest()[:20]

            audio_bytes = st.session_state.tts_audio.get(
                answer_id
            )

            if audio_bytes:

                should_autoplay = (
                    st.session_state.get("autoplay_audio_id")
                    == answer_id
                )

                st.audio(
                    audio_bytes,
                    format="audio/wav",
                    autoplay=should_autoplay,
                )

                if should_autoplay:
                    st.session_state.autoplay_audio_id = None

            else:
                voice_error = st.session_state.tts_errors.get(
                    answer_id
                )

                if voice_error:
                    st.caption(
                        "Voice could not be prepared. "
                        "The text answer is still available."
                    )
                else:
                    st.caption(
                        "Voice is not available for this answer yet."
                    )


# =========================================================
# 17. CHAT INPUT: TEXT + MICROPHONE + SEND
# =========================================================

st.divider()

st.caption(
    "Type a question or press 🎙️ to record. "
    "Stop recording to submit your voice question."
)

text_col, mic_col, send_col = st.columns(
    [12, 1, 1],
    vertical_alignment="bottom",
)

with text_col:

    typed_question = st.text_input(
        "Message",
        placeholder=(
            "Ask something about your PDF..."
            if st.session_state.active_mode == "PDF"
            else "Ask me anything..."
        ),
        label_visibility="collapsed",
        key="chat_message_input",
    )

with mic_col:

    voice_result = mic_recorder(
        start_prompt="🎙️",
        stop_prompt="⏹️",
        just_once=True,
        use_container_width=True,
        format="wav",
        key="chat_mic_recorder",
    )

with send_col:

    send_text = st.button(
        "➤",
        help="Send message",
        use_container_width=True,
        key="chat_send_button",
    )


# =========================================================
# 18. HANDLE TYPED QUESTION
# =========================================================

if send_text and typed_question.strip():

    submitted = submit_question(typed_question)

    if submitted:
        st.rerun()


# =========================================================
# 19. HANDLE VOICE QUESTION AUTOMATICALLY
# =========================================================

if voice_result and voice_result.get("bytes"):

    voice_bytes = voice_result["bytes"]

    voice_hash = hashlib.sha256(
        voice_bytes
    ).hexdigest()

    if st.session_state.last_voice_hash != voice_hash:

        st.session_state.last_voice_hash = voice_hash

        try:
            with st.spinner(
                "Voice-ai text-ah convert panni answer prepare pannudhu..."
            ):

                voice_question = transcribe_audio(
                    voice_bytes,
                    "audio/wav",
                )

                st.session_state.last_transcript = voice_question

                st.info(
                    f"🎤 You said: {voice_question}"
                )

                submitted = submit_question(
                    voice_question
                )

            if submitted:
                st.rerun()

        except errors.APIError as error:

            st.error(
                "Voice transcription API request failed."
            )

            if error.code == 429:
                st.warning(
                    "API quota/rate limit exceeded. "
                    "Try again later or check your Gemini API quota."
                )

            elif error.code == 503:
                st.warning(
                    "Gemini server is busy. "
                    "Please record again after a short wait."
                )

            else:
                st.caption(str(error))

        except Exception as error:

            st.error(
                "Voice-ai text-ah convert panna mudiyala."
            )

            st.caption(str(error))