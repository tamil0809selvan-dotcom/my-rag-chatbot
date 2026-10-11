
import os
import io
import time
import hashlib
import wave

import streamlit as st
import faiss
import numpy as np

from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors, types


# 1. CONFIGURATION
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)


# 2. SESSION STATE
DEFAULTS = {
    "active_mode": None,
    "messages": [],
    "pdf_hash": None,
    "pdf_name": None,
    "pdf_ready": False,
    "chunks": [],
    "faiss_index": None,
    "embedding_model": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

client = genai.Client(api_key=API_KEY) if API_KEY else None


# 3. EMBEDDING MODEL
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 4. GEMINI TEXT RESPONSE
def ask_gemini_with_retry(prompt, max_retries=3):
    if client is None:
        raise ValueError(
            "GEMINI_API_KEY missing. Check your .env file."
        )

    retryable_codes = {500, 502, 503, 504}

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

            return response.text

        except errors.APIError as error:
            if error.code not in retryable_codes:
                raise

            if attempt == max_retries - 1:
                raise

            time.sleep(2 ** attempt)

    raise RuntimeError("Gemini request failed.")


# 5. VOICE INPUT: SPEECH TO TEXT
def transcribe_audio(audio_bytes, mime_type="audio/wav"):
    if client is None:
        raise ValueError("Check your GEMINI_API_KEY in .env.")

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


# 6. VOICE OUTPUT: TEXT TO SPEECH
def text_to_speech(text):
    if client is None:
        raise ValueError("Check your GEMINI_API_KEY in .env.")

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=(
            "Read the following assistant response naturally "
            "and clearly. Use the language of the text. "
            "If it contains Tamil, speak Tamil; if it contains "
            "English, speak English. Do not translate it.\n\n"
            + text[:4000]
        ),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            ),
        ),
    )

    audio_data = None

    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                audio_data = part.inline_data.data
                break

    if not audio_data:
        raise ValueError("Gemini did not return audio.")

    # Wrap raw PCM audio in a WAV container.
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_data)

    return wav_buffer.getvalue()


# 7. PDF PROCESSING
def process_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            text_parts.append(
                f"[Page {page_number}]\n{page_text}"
            )

    pdf_text = "\n\n".join(text_parts).strip()

    if not pdf_text:
        raise ValueError(
            "No readable text found. Scanned PDFs may require OCR."
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

    embeddings = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, model


# 8. PDF RETRIEVAL
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
        query_embedding, dtype=np.float32
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


# 9. CHAT HISTORY
def build_history():
    history = st.session_state.messages[:-1][-10:]
    lines = []

    for message in history:
        role = (
            "User" if message["role"] == "user"
            else "Assistant"
        )
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)


# 10. PROMPT CREATION
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


# 11. NAVIGATION
def go_home():
    st.session_state.active_mode = None
    st.rerun()


def start_new_chat():
    st.session_state.messages = []
    st.rerun()


# 12. HOME PAGE
if st.session_state.active_mode is None:
    st.title("🤖 Advanced RAG Chatbot")

    st.write("Vanakkam bro! Welcome to your AI chatbot.")
    st.write("Choose how you want to chat.")

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

    st.caption("Tamil, Tanglish and English supported.")
    st.stop()


# 13. PDF UPLOAD
if (
    st.session_state.active_mode == "PDF"
    and not st.session_state.pdf_ready
):
    st.title("📄 PDF Mode")
    st.write("Upload your PDF to start chatting.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if st.button(
        "Process PDF & Open Chat",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        try:
            pdf_bytes = uploaded_file.getvalue()
            current_hash = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            with st.spinner("PDF process pannudhu..."):
                if (
                    current_hash == st.session_state.pdf_hash
                    and st.session_state.faiss_index is not None
                ):
                    st.session_state.pdf_name = uploaded_file.name
                else:
                    chunks, index, model = process_pdf(pdf_bytes)

                    st.session_state.chunks = chunks
                    st.session_state.faiss_index = index
                    st.session_state.embedding_model = model
                    st.session_state.pdf_hash = current_hash
                    st.session_state.pdf_name = uploaded_file.name

                st.session_state.messages = []
                st.session_state.pdf_ready = True

            st.success("PDF processed successfully!")
            st.rerun()

        except Exception as error:
            st.error("PDF process panna mudiyala.")
            st.caption(str(error))

    if st.button("⬅️ Back to Home"):
        go_home()

    st.stop()


# 14. CHAT HEADER
if st.session_state.active_mode == "PDF":
    st.title("📄 PDF Chat")
    st.caption(f"Document: {st.session_state.pdf_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅️ Home"):
            go_home()

    with col2:
        if st.button("🔄 Change PDF"):
            st.session_state.pdf_ready = False
            st.session_state.messages = []
            st.rerun()

    with col3:
        if st.button("🗑️ New Chat"):
            start_new_chat()

else:
    st.title("🧠 General AI Chat")
    st.caption(
        "Ask questions in Tamil, Tanglish or English."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅️ Home"):
            go_home()

    with col2:
        if st.button("🗑️ New Chat"):
            start_new_chat()


# 15. VOICE SETTINGS
st.divider()

st.caption(
    "Answers appear as text. Click 🔊 Listen below an answer "
    "to hear it in voice."
)

st.caption(
    "Use the microphone to speak, or type your question below."
)


# 16. VOICE INPUT
st.subheader("🎤 Voice Input")

audio_value = st.audio_input(
    "Record your question in Tamil, Tanglish or English",
    key="voice_question",
)

voice_button = st.button(
    "🎙️ Transcribe Voice & Ask",
    use_container_width=True,
    key="transcribe_and_ask",
)

voice_question = None

if voice_button:
    if audio_value is None:
        st.warning(
            "First microphone use panni question record pannu."
        )
    else:
        try:
            with st.spinner("Voice-ai text-ah convert pannudhu..."):
                voice_question = transcribe_audio(
                    audio_value.getvalue(),
                    audio_value.type or "audio/wav",
                )

            st.success("Voice converted to text!")
            st.write("**Your question:**")
            st.write(voice_question)

        except errors.APIError as error:
            st.error("Voice transcription API request failed.")

            if error.code == 429:
                st.warning(
                    "API quota/rate limit exceeded. "
                    "Try again later or check your Gemini API quota."
                )
            else:
                st.caption(str(error))

        except Exception as error:
            st.error("Voice-ai text-ah convert panna mudiyala.")
            st.caption(str(error))


# 17. DISPLAY EXISTING CHAT + SPEAKER BUTTON
st.divider()
st.subheader("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show speaker button only for AI answers.
        # Audio is generated only when the button is clicked.
        if message["role"] == "assistant":
            answer = message["content"]

            answer_id = hashlib.sha256(
                answer.encode("utf-8")
            ).hexdigest()[:16]

            audio_key = f"tts_audio_{answer_id}"

            if st.button(
                "🔊 Listen",
                key=f"listen_{answer_id}",
            ):
                try:
                    with st.spinner("Generating voice..."):
                        audio_bytes = text_to_speech(answer)

                    st.session_state[audio_key] = audio_bytes

                except Exception as error:
                    st.error("Voice output failed.")
                    st.caption(str(error))

            # Display audio player only after clicking Listen.
            if audio_key in st.session_state:
                st.audio(
                    st.session_state[audio_key],
                    format="audio/wav",
                )


# 18. TEXT INPUT OR VOICE QUESTION
typed_question = st.chat_input("Un question-ah kelu...")

# A newly transcribed voice question takes priority on that run.
question = voice_question or typed_question

if question:
    if client is None:
        st.error("GEMINI_API_KEY-ai .env file-la check pannu.")
        st.stop()

    if (
        st.session_state.active_mode == "PDF"
        and not st.session_state.pdf_ready
    ):
        st.warning("First PDF upload panni process pannu.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.chat_message("assistant"):
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

            # Show text answer first. No automatic voice generation.
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

        # Rerun to display the Listen button below the saved answer.
        st.rerun()

    except errors.APIError as error:
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and st.session_state.messages[-1]["content"] == question
        ):
            st.session_state.messages.pop()

        if error.code == 429:
            st.error(
                "Gemini API quota exceeded. Check your API quota "
                "and try again later."
            )
        elif error.code in {401, 403}:
            st.error(
                "Gemini API authentication failed. "
                "Check your API key and permissions."
            )
        else:
            st.error("Gemini API request failed.")
            st.caption(str(error))

    except Exception as error:
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and st.session_state.messages[-1]["content"] == question
        ):
            st.session_state.messages.pop()

        st.error("Chat request fail aayiduchu.")
        st.caption(str(error))
