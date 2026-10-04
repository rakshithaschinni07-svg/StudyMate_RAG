import streamlit as st
import re
import hashlib
import json
import math
import os
from collections import Counter

from pypdf import PdfReader
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StudyMate RAG",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 20px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        background-color: #f5f7fb;
        border: 1px solid #e1e5ee;
        margin-bottom: 15px;
        color : #222;
    }

    .source-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #4CAF50;
        margin-top: 10px;
        color : #222;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GEMINI SETUP
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"


# =========================================================
# SESSION STATE
# =========================================================

if "study_history" not in st.session_state:
    st.session_state.study_history = []

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_current_index" not in st.session_state:
    st.session_state.quiz_current_index = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False

if "quiz_selected_answer" not in st.session_state:
    st.session_state.quiz_selected_answer = None

if "quiz_round" not in st.session_state:
    st.session_state.quiz_round = 0

if "quiz_previous_questions" not in st.session_state:
    st.session_state.quiz_previous_questions = []

if "quiz_pdf_id" not in st.session_state:
    st.session_state.quiz_pdf_id = None


# =========================================================
# HISTORY
# =========================================================

def add_history(activity_type, title, content):
    st.session_state.study_history.append(
        {
            "type": activity_type,
            "title": title,
            "content": content
        }
    )


# =========================================================
# QUIZ RESET
# =========================================================

def reset_quiz():
    st.session_state.quiz_questions = []
    st.session_state.quiz_current_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_selected_answer = None
    st.session_state.quiz_round = 0
    st.session_state.quiz_previous_questions = []


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    full_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n\n"

    return full_text


# =========================================================
# PARAGRAPH-AWARE CHUNKING
# =========================================================

def create_chunks(text, max_chars=1200, overlap_paragraphs=1):

    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = []

    for paragraph in paragraphs:

        current_text = "\n\n".join(current_chunk)

        if (
            current_chunk
            and len(current_text) + len(paragraph) > max_chars
        ):

            chunks.append(current_text)

            current_chunk = current_chunk[-overlap_paragraphs:]

        current_chunk.append(paragraph)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# =========================================================
# LOCAL TEXT PROCESSING
# =========================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    stop_words = {
        "the",
        "is",
        "are",
        "was",
        "were",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "as",
        "at",
        "from",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "be",
        "can",
        "may",
        "into",
        "which",
        "what",
        "why",
        "how",
        "when",
        "where",
        "their",
        "they",
        "them",
        "than",
        "also",
        "such"
    }

    return [
        word
        for word in words
        if word not in stop_words and len(word) > 1
    ]


# =========================================================
# TF-IDF RETRIEVAL
# =========================================================

def create_tfidf_index(chunks):

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in chunks
    ]

    document_frequency = Counter()

    for tokens in tokenized_chunks:

        unique_words = set(tokens)

        for word in unique_words:
            document_frequency[word] += 1

    total_documents = len(chunks)

    return {
        "tokenized_chunks": tokenized_chunks,
        "document_frequency": document_frequency,
        "total_documents": total_documents
    }


def calculate_tfidf_vector(tokens, document_frequency, total_documents):

    term_frequency = Counter(tokens)

    vector = {}

    total_words = len(tokens)

    if total_words == 0:
        return vector

    for word, count in term_frequency.items():

        tf = count / total_words

        df = document_frequency.get(word, 0)

        if df == 0:
            continue

        idf = math.log(
            (total_documents + 1)
            /
            (df + 1)
        ) + 1

        vector[word] = tf * idf

    return vector


def cosine_similarity(vector_a, vector_b):

    if not vector_a or not vector_b:
        return 0.0

    common_words = set(vector_a.keys()) & set(vector_b.keys())

    dot_product = sum(
        vector_a[word] * vector_b[word]
        for word in common_words
    )

    magnitude_a = math.sqrt(
        sum(value * value for value in vector_a.values())
    )

    magnitude_b = math.sqrt(
        sum(value * value for value in vector_b.values())
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def retrieve_relevant_chunks(
    question,
    chunks,
    tfidf_index,
    top_k=5
):

    question_tokens = tokenize(question)

    question_vector = calculate_tfidf_vector(
        question_tokens,
        tfidf_index["document_frequency"],
        tfidf_index["total_documents"]
    )

    similarities = []

    for index, tokens in enumerate(
        tfidf_index["tokenized_chunks"]
    ):

        chunk_vector = calculate_tfidf_vector(
            tokens,
            tfidf_index["document_frequency"],
            tfidf_index["total_documents"]
        )

        similarity = cosine_similarity(
            question_vector,
            chunk_vector
        )

        similarities.append(
            (index, similarity)
        )

    similarities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_results = similarities[:top_k]

    return [
        {
            "index": index,
            "text": chunks[index],
            "similarity": similarity
        }
        for index, similarity in top_results
        if similarity > 0
    ]


# =========================================================
# GEMINI RESPONSE
# =========================================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:

        return (
            "I could not find the answer in the uploaded notes."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE CHUNK {item['index'] + 1}:

{item['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are StudyMate, an AI study assistant.

Answer the student's question using ONLY the uploaded study material
provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make assumptions.
4. If the answer cannot be found in the provided material, say exactly:

"I could not find the answer in the uploaded notes."

5. Use simple BCA-level language.
6. Keep the answer focused on the question.
7. If there are multiple points, use bullet points.
8. If the question asks for a comparison, use a clear comparison.
9. Do not mention that you are using a retrieval system.

UPLOADED STUDY MATERIAL:

{context}

STUDENT QUESTION:

{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(full_text):

    prompt = f"""
Create an exam-friendly summary of the uploaded study material.

IMPORTANT:

- Use ONLY the uploaded material.
- Do not add outside information.
- Do not invent facts.
- Cover the important topics.
- Use simple BCA-level English.
- Use headings and bullet points.
- Include important definitions and concepts.
- Make it useful for exam revision.

UPLOADED MATERIAL:

{full_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# =========================================================
# QUIZ GENERATION
# =========================================================

def generate_quiz(
    full_text,
    previous_questions
):

    previous_text = "\n".join(
        previous_questions
    )

    prompt = f"""
Create exactly 5 multiple-choice questions from the uploaded study
material.

Rules:

1. Use ONLY the uploaded material.
2. Do not use outside knowledge.
3. Do not repeat previous questions.
4. Try to test different concepts.
5. Each question must have exactly four options:
   A
   B
   C
   D
6. Only one option must be correct.
7. Include a short explanation.
8. Questions should be suitable for a BCA student.
9. Return ONLY valid JSON.
10. Do not use Markdown.

JSON format:

[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "A",
    "explanation": "Short explanation"
  }}
]

PREVIOUS QUESTIONS:

{previous_text}

UPLOADED STUDY MATERIAL:

{full_text}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    raw_text = response.text.strip()

    raw_text = re.sub(
        r"^```json\s*",
        "",
        raw_text
    )

    raw_text = re.sub(
        r"\s*```$",
        "",
        raw_text
    )

    try:

        questions = json.loads(raw_text)

    except json.JSONDecodeError:

        return []

    if not isinstance(questions, list):
        return []

    if len(questions) != 5:
        return []

    valid_questions = []

    for question in questions:

        if not isinstance(question, dict):
            continue

        if "question" not in question:
            continue

        if "options" not in question:
            continue

        if "correct_answer" not in question:
            continue

        if "explanation" not in question:
            continue

        options = question["options"]

        if not isinstance(options, dict):
            continue

        if set(options.keys()) != {"A", "B", "C", "D"}:
            continue

        if question["correct_answer"] not in {
            "A",
            "B",
            "C",
            "D"
        }:
            continue

        valid_questions.append(question)

    if len(valid_questions) != 5:
        return []

    return valid_questions


# =========================================================
# STUDY WITHOUT PDF
# =========================================================

def generate_topic_lesson(
    subject,
    topic,
    study_type
):

    language_instruction = ""

    subject_lower = subject.lower()

    if "kannada" in subject_lower:

        language_instruction = """
The subject is Kannada.

Write the entire answer in Kannada.

Use Kannada headings and Kannada explanations.

Do not write English paragraphs.

English may be used only for unavoidable technical terms.
"""

    elif "hindi" in subject_lower:

        language_instruction = """
The subject is Hindi.

Write the entire answer in Hindi.

Use Hindi headings and Hindi explanations.
"""

    elif "english" in subject_lower:

        language_instruction = """
The subject is English.

Write the answer in simple English.
"""

    else:

        language_instruction = """
This is a technical or general academic subject.

Write the answer in simple English suitable for a BCA student.
"""

    prompt = f"""
You are an academic study assistant.

Subject:
{subject}

Topic:
{topic}

Study type:
{study_type}

{language_instruction}

IMPORTANT ACCURACY RULES:

1. Do not pretend that you have the student's textbook.
2. Do not pretend that you have their syllabus.
3. Do not invent an author, poem, story, historical context, quotation,
or textbook-specific meaning.
4. If this is a specific literature lesson and exact context is needed,
clearly say that the textbook or lesson text is required.
5. Do not invent quotations.
6. Give only information you can provide reliably.
7. Use simple student-friendly language.
8. Make the answer useful for exams.

Generate the requested study material.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📚 StudyMate")

    st.divider()

    st.subheader("📊 Study History")

    if st.session_state.study_history:

        st.write(
            f"{len(st.session_state.study_history)} "
            "activities in this session"
        )

        for activity in reversed(
            st.session_state.study_history
        ):

            st.write(
                f"{activity['type']} "
                f"**{activity['title']}**"
            )

    else:

        st.write(
            "No study activities yet."
        )

    st.divider()

    st.subheader("ℹ️ About StudyMate")

    st.write(
        """
StudyMate is an AI-powered study assistant
that helps students learn from their own
PDF notes and study topics.
"""
    )

    st.write(
        """
**RAG Flow**

PDF → Text → Chunks → Local Retrieval → Gemini → Answer
"""
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📚 StudyMate RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your AI-powered study assistant'
    '</div>',
    unsafe_allow_html=True
)

st.success(
    "🎓 Learn smarter. Revise faster."
)

st.write(
    "Upload your study material, ask questions, "
    "generate summaries, practice quizzes, or study a topic without a PDF."
)


# =========================================================
# TABS
# =========================================================

tab_pdf, tab_without_pdf = st.tabs(
    [
        "📄 Study From PDF",
        "📚 Study Without PDF"
    ]
)


# =========================================================
# PDF TAB
# =========================================================

with tab_pdf:

    st.subheader("📄 Upload Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        current_pdf_id = hashlib.md5(
            file_bytes
        ).hexdigest()

        if (
            st.session_state.quiz_pdf_id is not None
            and
            st.session_state.quiz_pdf_id != current_pdf_id
        ):

            reset_quiz()

            st.info(
                "📄 New PDF detected. "
                "The previous quiz has been reset."
            )

        st.session_state.quiz_pdf_id = current_pdf_id

        # -------------------------------------------------
        # PROCESS PDF
        # -------------------------------------------------

        if (
            "processed_pdf_id"
            not in st.session_state
            or
            st.session_state.processed_pdf_id
            != current_pdf_id
        ):

            with st.spinner(
                "Preparing your study material..."
            ):

                try:

                    full_text = extract_text_from_pdf(
                        uploaded_file
                    )

                    chunks = create_chunks(
                        full_text
                    )

                    tfidf_index = create_tfidf_index(
                        chunks
                    )

                    st.session_state.full_text = full_text
                    st.session_state.chunks = chunks
                    st.session_state.tfidf_index = tfidf_index
                    st.session_state.processed_pdf_id = current_pdf_id

                except Exception as e:

                    st.error(
                        f"Could not process the PDF: {e}"
                    )

                    st.stop()

        else:

            full_text = st.session_state.full_text
            chunks = st.session_state.chunks
            tfidf_index = st.session_state.tfidf_index


        # -------------------------------------------------
        # PDF INFORMATION
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="info-card">
            <b>📄 File:</b> {uploaded_file.name}<br>
            <b>📑 Pages:</b> {len(PdfReader(uploaded_file).pages)}<br>
            <b>🧩 Chunks:</b> {len(chunks)}<br>
            <b>📝 Characters:</b> {len(full_text)}
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

        feature_tabs = st.tabs(
            [
                "💬 Q&A",
                "📝 Summary",
                "🎯 Quiz",
                "📖 Document Explorer"
            ]
        )


        # =================================================
        # Q&A
        # =================================================

        with feature_tabs[0]:

            st.subheader(
                "💬 Ask Questions From Your Notes"
            )

            question = st.text_input(
                "Enter your question",
                placeholder="Example: What are the advantages of hash file organization?"
            )

            if st.button(
                "🔎 Ask Question",
                key="ask_question"
            ):

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Finding the relevant information..."
                    ):

                        try:

                            retrieved_chunks = retrieve_relevant_chunks(
                                question,
                                chunks,
                                tfidf_index,
                                top_k=5
                            )

                            answer = generate_answer(
                                question,
                                retrieved_chunks
                            )

                            st.markdown("### Answer")

                            st.write(answer)

                            add_history(
                                "💬 Question",
                                question,
                                answer
                            )

                            if retrieved_chunks:

                                st.markdown(
                                    "### 📚 Sources"
                                )

                                for item in retrieved_chunks:

                                    st.markdown(
                                        f"""
                                        <div class="source-card">
                                        <b>Chunk {item['index'] + 1}</b>
                                        &nbsp; | &nbsp;
                                        Similarity:
                                        {item['similarity']:.3f}
                                        <br><br>
                                        {item['text'][:500]}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                        except Exception as e:

                            st.error(
                                f"Could not generate the answer: {e}"
                            )


        # =================================================
        # SUMMARY
        # =================================================

        with feature_tabs[1]:

            st.subheader(
                "📝 Generate Study Summary"
            )

            if st.button(
                "📝 Generate Summary",
                key="generate_summary"
            ):

                with st.spinner(
                    "Creating your summary..."
                ):

                    try:

                        summary = generate_summary(
                            full_text
                        )

                        st.markdown(
                            "### 📚 Study Summary"
                        )

                        st.write(summary)

                        add_history(
                            "📝 Summary",
                            "PDF Summary",
                            summary
                        )

                    except Exception as e:

                        st.error(
                            f"Could not generate summary: {e}"
                        )


        # =================================================
        # QUIZ
        # =================================================

        with feature_tabs[2]:

            st.subheader(
                "🎯 Interactive Quiz"
            )

            # Generate first round
            if not st.session_state.quiz_questions:

                if st.button(
                    "🎯 Start Quiz",
                    key="start_quiz"
                ):

                    with st.spinner(
                        "Creating your quiz..."
                    ):

                        try:

                            questions = generate_quiz(
                                full_text,
                                st.session_state.quiz_previous_questions
                            )

                            if not questions:

                                st.error(
                                    "Could not create a valid quiz. Please try again."
                                )

                            else:

                                st.session_state.quiz_questions = questions
                                st.session_state.quiz_current_index = 0
                                st.session_state.quiz_score = 0
                                st.session_state.quiz_answered = False
                                st.session_state.quiz_selected_answer = None
                                st.session_state.quiz_round += 1

                                st.rerun()

                        except Exception as e:

                            st.error(
                                f"Could not generate quiz: {e}"
                            )


            # Quiz in progress
            elif (
                st.session_state.quiz_current_index
                <
                len(st.session_state.quiz_questions)
            ):

                current_index = (
                    st.session_state.quiz_current_index
                )

                current_question = (
                    st.session_state.quiz_questions[
                        current_index
                    ]
                )

                st.write(
                    f"### Question {current_index + 1} of 5"
                )

                st.write(
                    current_question["question"]
                )

                selected_answer = st.radio(
                    "Choose your answer:",
                    options=[
                        "A",
                        "B",
                        "C",
                        "D"
                    ],
                    format_func=lambda option:
                        f"{option}. {current_question['options'][option]}",
                    key=f"quiz_option_{current_index}"
                )

                if not st.session_state.quiz_answered:

                    if st.button(
                        "✅ Submit Answer",
                        key=f"submit_{current_index}"
                    ):

                        st.session_state.quiz_selected_answer = selected_answer
                        st.session_state.quiz_answered = True

                        if (
                            selected_answer
                            ==
                            current_question["correct_answer"]
                        ):

                            st.session_state.quiz_score += 1

                        st.rerun()

                else:

                    correct_answer = (
                        current_question[
                            "correct_answer"
                        ]
                    )

                    if (
                        st.session_state.quiz_selected_answer
                        ==
                        correct_answer
                    ):

                        st.success(
                            "✅ Correct!"
                        )

                    else:

                        st.error(
                            "❌ Incorrect!"
                        )

                        st.info(
                            f"Correct answer: "
                            f"**{correct_answer}. "
                            f"{current_question['options'][correct_answer]}**"
                        )

                    st.write(
                        f"**Explanation:** "
                        f"{current_question['explanation']}"
                    )

                    if st.button(
                        "➡️ Next Question",
                        key=f"next_{current_index}"
                    ):

                        st.session_state.quiz_current_index += 1
                        st.session_state.quiz_answered = False
                        st.session_state.quiz_selected_answer = None

                        st.rerun()


            # Quiz completed
            else:

                score = st.session_state.quiz_score

                percentage = (
                    score / 5
                ) * 100

                st.success(
                    f"🎉 Quiz Complete!"
                )

                st.write(
                    f"### Score: {score}/5"
                )

                st.write(
                    f"### Percentage: {percentage:.0f}%"
                )

                add_history(
                    "🎯 Quiz",
                    f"Quiz Round {st.session_state.quiz_round}",
                    f"Score: {score}/5 ({percentage:.0f}%)"
                )

                st.divider()

                if st.button(
                    "🔄 Next 5 Questions",
                    key="next_quiz_round"
                ):

                    for question in (
                        st.session_state.quiz_questions
                    ):

                        st.session_state.quiz_previous_questions.append(
                            question["question"]
                        )

                    st.session_state.quiz_questions = []
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answered = False
                    st.session_state.quiz_selected_answer = None

                    st.rerun()


        # =================================================
        # DOCUMENT EXPLORER
        # =================================================

        with feature_tabs[3]:

            st.subheader(
                "📖 Document Explorer"
            )

            st.write(
                f"This document contains "
                f"**{len(chunks)} chunks**."
            )

            selected_chunk = st.selectbox(
                "Select a chunk",
                range(len(chunks)),
                format_func=lambda x:
                    f"Chunk {x + 1}"
            )

            st.markdown(
                f"""
                <div class="source-card">
                <b>Chunk {selected_chunk + 1}</b>
                <br><br>
                {chunks[selected_chunk]}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# STUDY WITHOUT PDF
# =========================================================

with tab_without_pdf:

    st.subheader(
        "📚 Study Without a PDF"
    )

    st.write(
        "Don't have notes or a PDF? Enter your subject "
        "and topic, then choose what you want to learn."
    )

    st.warning(
        "For specific language or literature lessons, "
        "your textbook or PDF is recommended because "
        "the exact context matters."
    )

    subject = st.text_input(
        "Subject",
        placeholder="Example: DBMS / Kannada / English"
    )

    topic = st.text_input(
        "Topic / Lesson",
        placeholder="Example: File Organization"
    )

    study_type = st.selectbox(
        "What do you want to learn?",
        [
            "Simple Explanation",
            "Exam Preparation",
            "Important Points",
            "10-Mark Answer",
            "Quick Revision"
        ]
    )

    if st.button(
        "📚 Study Topic",
        key="study_topic"
    ):

        if not subject.strip():

            st.warning(
                "Please enter the subject."
            )

        elif not topic.strip():

            st.warning(
                "Please enter the topic."
            )

        else:

            with st.spinner(
                "Preparing your study material..."
            ):

                try:

                    lesson = generate_topic_lesson(
                        subject,
                        topic,
                        study_type
                    )

                    st.markdown(
                        "### 📚 Study Material"
                    )

                    st.write(lesson)

                    add_history(
                        "📚 Topic Study",
                        topic,
                        lesson
                    )

                except Exception as e:

                    st.error(
                        f"Could not generate study material: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "StudyMate RAG • Learn smarter. Revise faster. 🎓"
)