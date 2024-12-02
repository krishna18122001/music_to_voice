import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import fitz  # PyMuPDF for PDF text extraction
import sounddevice as sd
import wave
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    """Convert text to speech and play it."""
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Function to select the most relevant chunk
def select_relevant_chunk(chunks, question):
    similarity_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    
    question_embedding = np.mean(similarity_model(question[:512]), axis=1)
    chunk_embeddings = [np.mean(similarity_model(chunk[:512]), axis=1) for chunk in chunks]
    
    similarities = [cosine_similarity(question_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0] for chunk_embedding in chunk_embeddings]
    
    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index]

# Function to query the LLM with the most relevant chunk
def query_llm(text, question):
    chunks = split_text_into_chunks(text, max_length=256)
    relevant_chunk = select_relevant_chunk(chunks, question)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")

    input_text = f"Here is a relevant part of the PDF:\n\n{relevant_chunk}\n\nQuestion: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=100, num_return_sequences=1, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Function to recognize speech from audio file
def recognize_speech(audio_file):
    model = whisper.load_model("large")
    result = model.transcribe(audio_file)
    transcript = result["text"]

    # Check for common errors like "kg" or repetitive patterns
    if any(word in transcript for word in ["kg", "x4.5"]):
        st.warning("The transcription may have errors. Please check and correct manually.")
        st.write(f"Transcribed text: {transcript}")
    else:
        return transcript

# Function to record audio
def record_audio(filename, duration=5, fs=16000):
    st.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording)

# Streamlit application
st.title("Voice Q&A with PDF Content")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)  # Ensure the temp directory exists
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text from the uploaded PDF file
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(temp_file_path)
    st.success("Text extraction completed!")

    # Define the path for the permanent audio file
    permanent_audio_path = "permanent_audio.wav"

    # Record and transcribe voice input
    st.write("Click to record your voice question:")
    if st.button("Record"):
        record_audio(permanent_audio_path, duration=5, fs=16000)

        # Transcribe the recorded voice input to text
        with st.spinner("Transcribing your question..."):
            question = recognize_speech(permanent_audio_path)
        st.success("Transcription completed!")
        st.write(f"Your Question: {question}")

        if question:
            # Generate an answer using the LLM
            with st.spinner("Generating answer..."):
                answer = query_llm(extracted_text, question)
            st.success("Answer generated!")
            st.write(answer)

            # Convert the answer to speech
            st.write("Speaking the answer...")
            speak(answer)
