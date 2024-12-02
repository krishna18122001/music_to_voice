import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import fitz  # PyMuPDF for PDF text extraction
import sounddevice as sd
import numpy as np
import tempfile
import wavio

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

# Function to query the LLM
def query_llm(text, question):
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
    input_text = f"Here is the text from the PDF:\n\n{text}\n\nQuestion: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Question:")[1].strip() if "Question:" in response else response.strip()

# Function to convert speech to text using Whisper
def speech_to_text(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    return result["text"]

# Function to record audio from the microphone
def record_audio(duration=5, fs=44100):
    st.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait()  # Wait until recording is finished
    st.success("Recording complete!")
    return np.squeeze(recording)

# Save the recording to a temporary file
def save_audio(audio, fs, file_name="output.wav"):
    temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
    wavio.write(temp_file_path, audio, fs, sampwidth=2)
    return temp_file_path

# Streamlit application
st.title("PDF Text Extraction and Q&A with Voice Interaction")

# Step 1: Upload PDF file
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

    # Step 2: Record your question using the microphone
    st.header("Ask a question about the document using your voice")
    if st.button("Record Question"):
        audio = record_audio(duration=5)  # Record for 5 seconds
        audio_file_path = save_audio(audio, 44100)

        # Convert the speech to text
        with st.spinner("Converting speech to text..."):
            question = speech_to_text(audio_file_path)
        st.success(f"Question recognized: {question}")

        # Generate the answer
        with st.spinner("Generating answer..."):
            answer = query_llm(extracted_text, question)
        st.success("Answer generated!")
        st.write(answer)

        # Speak out the answer
        speak(answer)
