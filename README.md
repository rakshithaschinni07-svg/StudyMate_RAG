# 📚 StudyMate RAG

StudyMate RAG is an AI-powered study assistant that helps students learn from their own PDF notes and study topics without a PDF.

## ✨ Features

- 📄 Upload and process PDF study material
- 💬 Ask questions from uploaded notes
- 📝 Generate study summaries
- 🎯 Interactive quizzes
- 📖 Explore document chunks
- 📚 Study topics without a PDF
- 📊 Session-based study history
- 🔎 Retrieval-Augmented Generation (RAG)

## 🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- PyPDF
- RAG

## 🧠 How RAG Works

PDF → Text Extraction → Chunking → Embeddings → Similarity Search → Relevant Context → Gemini → Answer

## 🔐 Security

The Gemini API key is stored in Streamlit secrets and is not included in the GitHub repository.

## ▶️ Run Locally

```bash
streamlit run app.py