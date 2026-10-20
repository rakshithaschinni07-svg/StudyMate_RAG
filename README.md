# 📚 StudyMate RAG

**StudyMate RAG** is an AI-powered study assistant that helps students learn from their own PDF notes, prepare for exams, and explore study topics with or without uploaded documents.

## ✨ Features

- 📄 **PDF Upload and Processing** — Upload PDF notes and extract their text.
- 💬 **Question Answering** — Ask questions and receive answers based on relevant content from your notes.
- 🧠 **Retrieval-Augmented Generation (RAG)** — Retrieve relevant text chunks to provide context for AI-generated answers.
- 📝 **Study Summaries** — Generate concise summaries of study material.
- 🎯 **Interactive Quizzes** — Generate practice questions to test your understanding.
- 📖 **Document Explorer** — Explore extracted text chunks from uploaded documents.
- 📚 **Study Without PDF** — Learn about topics using AI-generated explanations.
- 📝 **Exam Preparation** — Generate exam-style answers and evaluate practice answers.
- 📂 **Previous Question Papers** — Manage question papers and access supported PDF sources.
- 🔁 **Repeated Question Analysis** — Identify recurring questions across available question papers.
- 📊 **Study History** — Review study activities during the current session or through supported saved history features.

## 🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- PyPDF
- SQLite
- Retrieval-Augmented Generation (RAG)

## 🧠 How RAG Works

StudyMate RAG uses the following workflow:

**PDF → Text Extraction → Text Chunking → Embeddings → Similarity Search → Relevant Context → Gemini → Answer**

1. **Text Extraction:** Extract readable text from uploaded PDF files.
2. **Chunking:** Divide extracted text into smaller sections.
3. **Embeddings:** Represent text chunks numerically for semantic comparison.
4. **Similarity Search:** Retrieve chunks relevant to the user's question.
5. **Contextual Answering:** Send the retrieved context to Gemini to generate a response.

Answer quality depends on the quality of the extracted text and the relevance of the retrieved chunks. Scanned PDFs may require OCR.

## 🔐 Security and Configuration

- Store the Gemini API key in Streamlit secrets or environment configuration.
- Never commit API keys, passwords, or other secrets to GitHub.
- Keep database files and private study documents out of the public repository unless intentionally shared.
- Review external PDF downloads and URLs before using them.

For Streamlit Cloud, secrets can be configured through the app's settings. For Render, configure the required secret through the service's environment or secret settings, according to your application code.

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/rakshithaschinni07-svg/StudyMate_RAG.git
cd StudyMate_RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Add your API key using the configuration expected by the application. If using Streamlit secrets, create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

**Do not commit this file to GitHub.**

### 4. Run the application

If running the current version:

```bash
streamlit run app.py
```

If testing the updated version before deployment:

```bash
streamlit run app_new.py
```

## 🚀 Deployment

StudyMate RAG can be deployed as a Streamlit application on a compatible hosting platform such as Render. Configure the required API key and dependencies in the hosting service.

**Note:** Local files and SQLite databases on ephemeral hosting storage may not persist across restarts or redeployments. Use appropriate persistent storage or backups if study history and uploaded question papers must be retained.

## 🎓 Project Goal

StudyMate RAG aims to make studying more interactive and efficient by helping students understand notes, revise important concepts, practice quizzes, and prepare for exams.

**Learn smarter. Revise faster.**
