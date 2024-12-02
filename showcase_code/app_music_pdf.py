import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import PyPDF2

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        print(f"Text-to-speech error: {e}")

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_path, output_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcribed_text = result['text']
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(transcribed_text)
    return transcribed_text

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path, output_file_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    return text

# Function to query the LLM
def query_llm(transcription_text, question):
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
    input_text = f"Here is the transcription of the file:\n\n{transcription_text}\n\nQuestion: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Question:")[1].strip() if "Question:" in response else response.strip()

# Streamlit application
st.title("Transcription and Q&A with Whisper and LLM")

# Upload file
uploaded_file = st.file_uploader("Upload a file (MP3 or PDF)", type=["mp3", "pdf"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)  # Ensure the temp directory exists
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Define output file path for the transcription
    output_file_path = "temp/output.txt"

    # Process the uploaded file based on its type
    if uploaded_file.name.endswith(".mp3"):
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio_with_whisper(temp_file_path, output_file_path)
        st.success("Audio transcription completed!")
    elif uploaded_file.name.endswith(".pdf"):
        with st.spinner("Extracting text from PDF..."):
            transcribed_text = extract_text_from_pdf(temp_file_path, output_file_path)
        st.success("PDF text extraction completed!")
    
    # Ask a question about the transcription/extracted text
    question = st.text_input("Ask a question about the content")

    if question:
        with st.spinner("Generating answer..."):
            answer = query_llm(transcribed_text, question)
        st.success("Answer generated!")
        
        # Speak out the answer
        speak(answer)
