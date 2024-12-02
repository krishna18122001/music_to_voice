import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import fitz  # PyMuPDF for PDF text extraction
import sounddevice as sd
import tempfile
import wave
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

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

    # Truncate the question to match the maximum length
    question_embedding = np.mean(similarity_model(question[:512]), axis=1)

    chunk_embeddings = [np.mean(similarity_model(chunk[:512]), axis=1) for chunk in chunks]

    # Use cosine similarity to compare embeddings
    similarities = [cosine_similarity(question_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0] for chunk_embedding in chunk_embeddings]

    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index]

# Function to query the LLM with the most relevant chunk
def query_llm(text, question):

    print('text', text)

    print('question', question)

    chunks = split_text_into_chunks(text, max_length=256)  # Adjust chunk size here
    relevant_chunk = select_relevant_chunk(chunks, question)

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
    model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")

    # Ensure truncation in the tokenizer
    input_text = f"Here is a relevant part of the PDF:\n\n{relevant_chunk}\n\nQuestion: {question}"
    # input_text = f"Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_new_tokens=250, num_return_sequences=1, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('responseeee', response)
    # return response.strip()
    return response.split("Question:")[1].strip() if "Question:" in response else response.strip()

# Function to record audio with debug information
def record_audio(filename, duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        st.success("Recording completed.")
        
        # Debug: Check if audio was recorded
        if np.all(recording == 0):
            st.warning("No audio data recorded. Please check your microphone.")
            return None
        # Normalize the audio data
        recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())
        
        st.success(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        st.error(f"Error during audio recording: {str(e)}")
        return None

# Function to recognize speech from audio file with debug information
def recognize_speech(audio_file, save_path="transcribed_question.txt"):
    st.info("Loading Whisper model...")
    try:
        model = whisper.load_model("base")
        st.success("Whisper model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

    st.info("Transcribing audio...")
    try:
        result = model.transcribe(audio_file)
        transcript = result["text"]
        st.success("Transcription completed.")
        
        # Save the transcript to a file
        with open(save_path, "w") as f:
            f.write(transcript)
        st.success(f"Transcription saved to {save_path}")
        
        # Check for common errors like "kg" or repetitive patterns
        if any(word in transcript for word in ["kg", "x4.5"]):
            st.warning("The transcription may have errors. Please check and correct manually.")
        
        return transcript
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

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

    # Option to upload a WAV file for transcription
    st.write("Upload a WAV file or record your question:")
    uploaded_audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(uploaded_audio_file.read())
            audio_path = temp_audio_file.name

        # Transcribe the uploaded audio file
        with st.spinner("Transcribing your question..."):
            question = recognize_speech(audio_path, save_path="transcribed_question.txt")
        
        if question:
            st.success("Transcription completed and saved!")
            st.write(f"Your Question: {question}")

            # Generate an answer using the LLM
            with st.spinner("Generating answer..."):
                answer = query_llm(extracted_text, question)
            st.success("Answer generated!")
            st.write(answer)

            # Convert the answer to speech
            st.write("Speaking the answer...")
            speak(answer)
        else:
            st.error("Failed to transcribe the audio. Please try again.")
    else:
        # Option to record audio
        st.write("Or click to record your voice question:")
        if st.button("Record"):
            audio_file = "debug_audio.wav"
            audio_path = record_audio(audio_file)
            
            if audio_path:
                st.audio(audio_path, format='audio/wav')
                
                # Transcribe the recorded voice input to text and save it to a file
                with st.spinner("Transcribing your question..."):
                    question = recognize_speech(audio_path, save_path="transcribed_question.txt")
                
                if question:
                    st.success("Transcription completed and saved!")
                    st.write(f"Your Question: {question}")

                    # Generate an answer using the LLM
                    with st.spinner("Generating answer..."):
                        answer = query_llm(extracted_text, question)
                    st.success("Answer generated!")
                    st.write(answer)

                    # Convert the answer to speech
                    st.write("Speaking the answer...")
                    speak(answer)
                else:
                    st.error("Failed to transcribe the audio. Please try again.")
            else:
                st.error("Failed to record audio. Please check your microphone and try again.")
