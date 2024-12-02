import streamlit as st
import whisper
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import sounddevice as sd
import wave
import numpy as np
import PyPDF2
from moviepy.editor import VideoFileClip

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    try:
        engine.say(message)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

# Function to transcribe audio using Whisper
def transcribe_audio_with_whisper(audio_path, output_file_path):
    if os.path.exists(output_file_path):
        with open(output_file_path, 'r', encoding='utf-8') as file:
            return file.read()

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcribed_text = result['text']
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(transcribed_text)
    return transcribed_text

# Function to query the LLM
def query_llm(transcription_text, pdf_text, question):
    try:
        tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
        model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")
        input_text = f"Here is the transcription of the audio:\n\n{transcription_text}\n\nHere is the text from the PDF:\n\n{pdf_text}\n\nQuestion: {question}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        outputs = model.generate(inputs['input_ids'], max_new_tokens=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Question:")[1].strip() if "Question:" in response else response.strip()
    except Exception as e:
        st.error(f"Error querying LLM: {e}")
        return "An error occurred while querying the LLM."

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# Function to record audio
def record_audio(filename, duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        st.success("Recording completed.")
        
        # Normalize the audio data
        recording = np.int16(recording / np.max(np.abs(recording)) * 32767)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(recording.tobytes())
        
        return filename
    except Exception as e:
        st.error(f"Error during audio recording: {str(e)}")
        return None

# Function to recognize speech from audio file
def recognize_speech(audio_file):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        transcript = result["text"]
        return transcript
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""

# Function to extract audio from MP4
def extract_audio_from_mp4(mp4_file, audio_file):
    try:
        video = VideoFileClip(mp4_file)
        audio = video.audio
        audio.write_audiofile(audio_file)
    except Exception as e:
        st.error(f"Error extracting audio from MP4: {e}")

# Reset session state
def reset_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Streamlit application
st.title("Audio, PDF & MP4 Transcription and Q&A with Whisper and LLM")

# Add reset button
if st.button("Reset"):
    reset_session_state()

# Upload MP4 file
uploaded_mp4 = st.file_uploader("Upload an MP4 file", type=["mp4"], key="mp4")

# Upload audio file
uploaded_audio = st.file_uploader("Upload an MP3 file", type=["mp3"], key="audio")

# Upload PDF file
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf")

# Initialize session state variables
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'mp4_transcribed_text' not in st.session_state:
    st.session_state.mp4_transcribed_text = ""
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'question' not in st.session_state:
    st.session_state.question = ""

# Process files
if uploaded_mp4 or uploaded_audio or uploaded_pdf:
    if uploaded_mp4 is not None:
        try:
            temp_mp4_path = os.path.join("temp", uploaded_mp4.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_mp4_path, "wb") as f:
                f.write(uploaded_mp4.read())

            audio_file = "temp/mp4_audio.wav"
            extract_audio_from_mp4(temp_mp4_path, audio_file)
            output_audio_file_path = "temp/mp4_transcription.txt"
            st.session_state.mp4_transcribed_text = transcribe_audio_with_whisper(audio_file, output_audio_file_path)
            st.success("MP4 audio transcription completed!")
        except Exception as e:
            st.error(f"Error processing MP4 file: {e}")
    
    if uploaded_audio is not None:
        try:
            temp_audio_path = os.path.join("temp", uploaded_audio.name)
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_audio.read())

            output_audio_file_path = "temp/audio_transcription.txt"
            st.session_state.transcribed_text = transcribe_audio_with_whisper(temp_audio_path, output_audio_file_path)
            st.success("MP3 audio transcription completed!")
        except Exception as e:
            st.error(f"Error processing MP3 file: {e}")
    
    if uploaded_pdf is not None:
        try:
            with st.spinner("Extracting text from PDF..."):
                st.session_state.pdf_text = extract_text_from_pdf(uploaded_pdf)
            st.success("PDF text extraction completed!")
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
    
    if st.button("Record your voice question"):
        try:
            audio_file = "temp/voice_question.wav"
            audio_path = record_audio(audio_file)
            
            if audio_path:
                st.audio(audio_path, format='audio/wav')
                
                with st.spinner("Transcribing your question..."):
                    st.session_state.question = recognize_speech(audio_path)
                
                if st.session_state.question:
                    st.success("Transcription completed!")
                    st.write(f"Your Question: {st.session_state.question}")

                    with st.spinner("Generating answer..."):
                        answer = query_llm(st.session_state.transcribed_text + "\n" + st.session_state.mp4_transcribed_text, st.session_state.pdf_text, st.session_state.question)
                    st.success("Answer generated!")
                    st.write(answer)

                    st.write("Speaking the answer...")
                    speak(answer)
                else:
                    st.error("Failed to transcribe the audio. Please try again.")
            else:
                st.error("Failed to record audio. Please check your microphone and try again.")
        except Exception as e:
            st.error(f"Error recording or processing voice question: {e}")
else:
    st.info("Upload an MP4, MP3, or PDF file to get started.")
