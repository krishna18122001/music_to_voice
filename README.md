## Voice Q&A with PDF Content

This Streamlit-based application allows users to interact with a PDF document by asking questions using their voice. The application processes uploaded PDFs, transcribes spoken questions from audio files, and generates answers based on the content of the PDF. The response is provided in both text and spoken form.

### Features:
- **PDF Text Extraction**: Upload a PDF, and the application extracts the text using **PyMuPDF** (`fitz`).
- **Voice Input**: Ask questions by uploading a **WAV file** or recording your voice directly in the app. 
- **Speech-to-Text**: The application uses **Whisper** (by OpenAI) to transcribe spoken questions into text.
- **Relevance Matching**: The PDF text is split into chunks, and the most relevant section is identified using **Sentence-Transformers** and **cosine similarity**.
- **Question Answering**: **StableLM Zephyr** (from StabilityAI) generates an answer to the user's question using the extracted text.
- **Text-to-Speech**: The generated answer is read aloud using **pyttsx3**.

### Libraries and Tools Used:
- **Streamlit**: For the web interface and user interaction.
- **Whisper**: Speech-to-text model for transcribing audio.
- **PyMuPDF (fitz)**: For extracting text from PDF files.
- **Sentence Transformers**: For text embedding and relevance matching using cosine similarity.
- **StableLM Zephyr**: Large language model for question answering.
- **pyttsx3**: Text-to-speech engine for reading answers aloud.
- **sounddevice**: For recording audio input.
- **scikit-learn**: For cosine similarity calculation.

### How It Works:
1. Upload a PDF and extract its text.
2. Ask a question using voice input (upload or record).
3. The question is transcribed and matched with relevant text from the PDF.
4. The answer is generated using a large language model and read aloud.
