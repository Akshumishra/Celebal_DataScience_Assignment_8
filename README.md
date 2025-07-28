# RAG Chatbot with Gemini Pro

This project is a Retrieval-Augmented Generation (RAG) chatbot using Google Gemini Pro, Streamlit, and vector search with FAISS. It allows you to ask questions based on the content of PDF documents you provide.

## Features
- Upload and process PDF documents
- Vector search using FAISS and sentence-transformers
- Answer questions using Google Gemini Pro
- Streamlit web interface

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Celebal_DataScience_Assignment_8.git
cd Celebal_DataScience_Assignment_8
```

### 2. Install Dependencies
Install all required Python packages using:
```bash
pip install -r requirements.txt
```

### 3. Set Up Your Gemini API Key
Set your Gemini API key as an environment variable before running the app. In PowerShell (Windows):
```powershell
$env:GEMINI_API_KEY="your_actual_gemini_api_key"
```
Replace `your_actual_gemini_api_key` with your real Gemini API key.

### 4. Add Your PDF Documents
Create a folder named `documents` in the project root and add your PDF files there:
```powershell
mkdir documents
# Place your PDFs inside the 'documents' folder
```

### 5. Run the Application
Start the Streamlit app with:
```powershell
python -m streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

## File Structure
- `app.py` — Main Streamlit app
- `rag_gemini_bot.py` — Core logic for PDF processing, vector search, and Gemini integration
- `requirements.txt` — List of required Python packages
- `documents/` — Folder for your PDF files

## Notes
- Make sure your environment variable is set in the same terminal session where you run the app.
- If you encounter missing module errors, ensure all dependencies are installed.

## License
This project is for educational purposes.
