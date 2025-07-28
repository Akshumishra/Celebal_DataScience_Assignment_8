import os
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ✅ Secure API key configuration
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
genai.configure(api_key=api_key)

# ✅ Corrected: Removed 'generation_method'
# Use a valid Gemini model from the list_models() output
# Example: If list_models shows 'models/gemini-2.5-flash-lite'
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite") # Corrected line

# ✅ Sentence transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdfs(folder_path):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = " ".join([page.extract_text() or "" for page in reader.pages])
                    texts.append(text)
            except PyPDF2.errors.PdfReadError as e:
                print(f"Error reading PDF {file}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with {file}: {e}")
    return texts


def build_vector_store(texts):
    if not texts:
        print("No texts to build vector store.")
        return None, []
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts


def get_top_k_docs(query, index, all_texts, k=3):
    if index is None or not all_texts:
        return [] # Return empty list if no index or texts
    query_vector = embedder.encode([query])
    _, indices = index.search(query_vector, k)
    return [all_texts[i] for i in indices[0]]


def generate_answer_with_gemini(query, index, all_texts): # Renamed for clarity
    context_docs = get_top_k_docs(query, index, all_texts)
    context = "\n".join(context_docs)
    
    # Check if context is empty
    if not context.strip():
        return "No relevant context found to answer your question. Please try a different query."

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

# Example Usage (assuming you have a 'pdfs' folder with PDF files)
if __name__ == "__main__":
    pdf_folder = "pdfs" # Make sure this folder exists and contains PDFs
    
    # Create a dummy 'pdfs' folder and a dummy PDF for testing if it doesn't exist
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        # Create a dummy PDF using reportlab for demonstration
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(os.path.join(pdf_folder, "sample1.pdf"), pagesize=letter)
        c.drawString(100, 750, "This is a sample document about artificial intelligence.")
        c.drawString(100, 730, "AI is a rapidly evolving field.")
        c.save()

        c = canvas.Canvas(os.path.join(pdf_folder, "sample2.pdf"), pagesize=letter)
        c.drawString(100, 750, "Machine learning is a subset of AI.")
        c.drawString(100, 730, "Deep learning is a more advanced technique.")
        c.save()
        print(f"Created dummy PDF files in '{pdf_folder}' for testing.")


    print(f"Extracting text from PDFs in: {pdf_folder}")
    extracted_texts = extract_text_from_pdfs(pdf_folder)
    
    if extracted_texts:
        print(f"Found {len(extracted_texts)} documents. Building vector store...")
        vector_index, original_texts = build_vector_store(extracted_texts)

        if vector_index:
            query = "What is AI?"
            print(f"\nQuery: {query}")
            answer = generate_answer_with_gemini(query, vector_index, original_texts)
            print(f"Answer: {answer}")

            query2 = "Tell me about deep learning."
            print(f"\nQuery: {query2}")
            answer2 = generate_answer_with_gemini(query2, vector_index, original_texts)
            print(f"Answer: {answer2}")
        else:
            print("Failed to build vector store.")
    else:
        print("No text extracted from PDFs. Please ensure the folder exists and contains valid PDF files.")