import streamlit as st
import re
import hashlib
import json
import numpy as np
import os

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StudyMate RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SIMPLE CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #cbd5e1;
        margin-bottom: 20px;
    }

    .pdf-card {
        background-color: #ffffff;
        color: #111827;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 16px;
        margin: 15px 0;
    }

    .source-card {
        background-color: #f8fafc;
        color: #111827;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }

    .score-card {
        background-color: #172554;
        color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


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

if "quiz_history_saved" not in st.session_state:
    st.session_state.quiz_history_saved = False


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
# RESET QUIZ
# =========================================================

def reset_quiz():

    st.session_state.quiz_questions = []
    st.session_state.quiz_current_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_answered = False
    st.session_state.quiz_selected_answer = None
    st.session_state.quiz_round = 0
    st.session_state.quiz_previous_questions = []
    st.session_state.quiz_history_saved = False


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages), len(reader.pages)


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(
    text,
    max_chars=1200,
    overlap_paragraphs=1
):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    chunks = []

    current_chunk = []

    for paragraph in paragraphs:

        current_text = "\n\n".join(
            current_chunk
        )

        if (
            current_chunk
            and
            len(current_text) + len(paragraph)
            > max_chars
        ):

            chunks.append(current_text)

            current_chunk = current_chunk[
                -overlap_paragraphs:
            ]

        current_chunk.append(paragraph)

    if current_chunk:

        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embeddings(chunks):

    return embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    embeddings
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )[0]

    similarities = np.dot(
        embeddings,
        question_embedding
    ) / (
        np.linalg.norm(
            embeddings,
            axis=1
        )
        *
        np.linalg.norm(
            question_embedding
        )
        + 1e-10
    )

    top_indices = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True
    )[:5]

    results = []

    for index in top_indices:

        results.append(
            {
                "chunk": chunks[index],
                "score": float(
                    similarities[index]
                ),
                "index": index
            }
        )

    return results


# =========================================================
# GEMINI RESPONSE
# =========================================================

def generate_response(prompt):

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# =========================================================
# QUESTION ANSWERING
# =========================================================

def answer_question(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        [
            f"Source {i + 1}:\n{item['chunk']}"
            for i, item in enumerate(
                retrieved_chunks
            )
        ]
    )

    prompt = f"""
You are StudyMate, an AI study assistant.

Answer the student's question using ONLY the uploaded study material.

Rules:

1. Use only the provided material.
2. Do not use outside facts.
3. Do not invent information.
4. If the answer is not available, say:

"I could not find the answer in the uploaded notes."

5. Use simple BCA-level English.
6. Use bullet points when useful.
7. For comparisons, give a clear comparison.
8. Stay focused on the question.

QUESTION:

{question}

UPLOADED STUDY MATERIAL:

{context}
"""

    return generate_response(prompt)


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(full_text):

    prompt = f"""
Create a simple exam-friendly summary of the uploaded study material.

Rules:

1. Use ONLY the uploaded material.
2. Do not add outside information.
3. Do not invent facts.
4. Cover important concepts.
5. Use simple BCA-level English.
6. Use headings and bullet points.
7. Include important definitions and comparisons.

UPLOADED MATERIAL:

{full_text}
"""

    return generate_response(prompt)


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
Create EXACTLY 5 multiple-choice questions
from the uploaded study material.

Rules:

1. Use ONLY the uploaded material.
2. Do not use outside knowledge.
3. Each question must have exactly 4 options.
4. Options must be A, B, C and D.
5. Exactly one option is correct.
6. Give a short explanation.
7. Use simple BCA-level English.
8. Cover different concepts.
9. Avoid repeating previous questions.
10. Return ONLY valid JSON.
11. Do not use Markdown.

JSON format:

[
  {{
    "question": "Question",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "A",
    "explanation": "Explanation"
  }}
]

PREVIOUS QUESTIONS:

{previous_text}

UPLOADED MATERIAL:

{full_text}
"""

    response = generate_response(
        prompt
    ).strip()

    if response.startswith("```"):

        response = re.sub(
            r"^```(?:json)?",
            "",
            response
        )

        response = re.sub(
            r"```$",
            "",
            response
        )

    data = json.loads(response)

    if not isinstance(data, list):
        raise ValueError("Invalid quiz format.")

    if len(data) != 5:
        raise ValueError(
            "Quiz must contain exactly 5 questions."
        )

    for question in data:

        if set(
            question["options"].keys()
        ) != {"A", "B", "C", "D"}:

            raise ValueError(
                "Invalid options."
            )

        if question["correct_answer"] not in {
            "A",
            "B",
            "C",
            "D"
        }:

            raise ValueError(
                "Invalid correct answer."
            )

    return data


# =========================================================
# STUDY WITHOUT PDF
# =========================================================

def generate_topic_lesson(
    subject,
    topic,
    study_type
):

    prompt = f"""
You are an educational assistant helping a BCA student.

Subject:
{subject}

Topic:
{topic}

Study type:
{study_type}

LANGUAGE RULES:

If the subject is Kannada:
- Write the entire answer in Kannada.
- Use Kannada headings.
- Do not write English paragraphs.
- English technical terms may be used only when necessary.

If the subject is Hindi:
- Write the entire answer in Hindi.

If the subject is English:
- Write in English.

For technical subjects:
- Use simple English.

ACCURACY RULES:

1. Do not pretend to have the student's textbook.
2. Do not claim the answer comes from uploaded notes.
3. Do not invent authors or quotations.
4. Do not invent literary context.
5. If exact textbook context is required, clearly say so.
6. Use simple student-friendly language.
7. Make the answer useful for exams.

Now prepare the requested study material.
"""

    return generate_response(prompt)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📚 StudyMate RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Your AI-powered study assistant for learning from your own notes or any topic.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# HERO
# IMPORTANT: NO HTML HERE
# =========================================================

st.success(
    "🎓 Learn smarter. Revise faster."
)

st.write(
    "Upload your study material to ask grounded questions, "
    "generate summaries, explore your document, and take "
    "interactive quizzes. You can also study any topic "
    "without a PDF."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📊 Study History")

    st.caption(
        "Your recent learning activity"
    )

    if not st.session_state.study_history:

        st.info(
            "No study activity yet."
        )

    else:

        st.write(
            f"{len(st.session_state.study_history)} "
            "activities in this session"
        )

        for item in reversed(
            st.session_state.study_history
        ):

            st.write(
                f"**{item['type']}**  \n"
                f"{item['title']}"
            )

            st.divider()

    st.markdown("## 📚 About StudyMate")

    st.write(
        "StudyMate uses Retrieval-Augmented Generation "
        "(RAG) to answer questions from your study material."
    )

    st.caption(
        "History is stored only during the current app session."
    )


# =========================================================
# MAIN MODE TABS
# =========================================================

pdf_mode, topic_mode = st.tabs(
    [
        "📄 Study From PDF",
        "📚 Study Without PDF"
    ]
)


# =========================================================
# STUDY FROM PDF
# =========================================================

with pdf_mode:

    st.header("📄 Study From PDF")

    st.write(
        "Upload your notes and learn directly from your study material."
    )

    uploaded_file = st.file_uploader(
        "Choose your study PDF",
        type=["pdf"]
    )

    if uploaded_file:

        file_bytes = uploaded_file.getvalue()

        current_pdf_id = hashlib.md5(
            file_bytes
        ).hexdigest()

        # Reset quiz when PDF changes

        if (
            st.session_state.quiz_pdf_id is not None
            and
            st.session_state.quiz_pdf_id
            != current_pdf_id
        ):

            reset_quiz()

            st.info(
                "📄 New PDF detected. "
                "The previous quiz has been reset."
            )

        st.session_state.quiz_pdf_id = current_pdf_id

        try:

            # -------------------------------------------------
            # PDF
            # -------------------------------------------------

            full_text, page_count = extract_text_from_pdf(
                uploaded_file
            )

            if not full_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()

            # -------------------------------------------------
            # CHUNKS
            # -------------------------------------------------

            chunks = create_chunks(
                full_text
            )

            # -------------------------------------------------
            # EMBEDDINGS
            # -------------------------------------------------

            with st.spinner(
                "Preparing your study material..."
            ):

                embeddings = create_embeddings(
                    chunks
                )

            # -------------------------------------------------
            # PDF INFORMATION
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="pdf-card">
                    <strong>📄 {uploaded_file.name}</strong>
                    <br><br>
                    <strong>Pages:</strong> {page_count}
                    &nbsp; • &nbsp;
                    <strong>Chunks:</strong> {len(chunks)}
                    &nbsp; • &nbsp;
                    <strong>Text:</strong> {len(full_text):,} characters
                </div>
                """,
                unsafe_allow_html=True
            )

            # -------------------------------------------------
            # FEATURE TABS
            # -------------------------------------------------

            qa_tab, summary_tab, quiz_tab, explorer_tab = st.tabs(
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

            with qa_tab:

                st.subheader(
                    "💬 Ask Your Notes"
                )

                question = st.text_input(
                    "Enter your question",
                    placeholder=(
                        "Example: What are the advantages "
                        "of Hash File Organization?"
                    )
                )

                if st.button(
                    "🔍 Get Answer",
                    use_container_width=True
                ):

                    if not question.strip():

                        st.warning(
                            "Please enter a question."
                        )

                    else:

                        with st.spinner(
                            "Searching your notes..."
                        ):

                            retrieved = retrieve_relevant_chunks(
                                question,
                                chunks,
                                embeddings
                            )

                            answer = answer_question(
                                question,
                                retrieved
                            )

                        st.subheader("💡 Answer")

                        st.write(answer)

                        add_history(
                            "💬 Question",
                            question,
                            answer
                        )

                        st.subheader(
                            "📚 Retrieved Sources"
                        )

                        for i, item in enumerate(
                            retrieved
                        ):

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <strong>
                                        Source {i + 1}
                                    </strong>
                                    <br>
                                    Similarity:
                                    {item['score']:.3f}
                                    <br><br>
                                    {item['chunk']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


            # =================================================
            # SUMMARY
            # =================================================

            with summary_tab:

                st.subheader(
                    "📝 Study Summary"
                )

                if st.button(
                    "✨ Generate Summary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Creating summary..."
                    ):

                        summary = generate_summary(
                            full_text
                        )

                    st.markdown(summary)

                    add_history(
                        "📝 Summary",
                        "PDF Summary",
                        summary
                    )


            # =================================================
            # QUIZ
            # =================================================

            with quiz_tab:

                st.subheader(
                    "🎯 Interactive Quiz"
                )

                st.write(
                    "Test your understanding with 5 questions."
                )

                # Start quiz

                if (
                    not st.session_state.quiz_questions
                    and
                    st.session_state.quiz_round == 0
                ):

                    if st.button(
                        "🚀 Start Quiz",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Generating 5 questions..."
                        ):

                            try:

                                questions = generate_quiz(
                                    full_text,
                                    st.session_state.quiz_previous_questions
                                )

                                st.session_state.quiz_questions = questions

                                st.session_state.quiz_current_index = 0

                                st.session_state.quiz_score = 0

                                st.session_state.quiz_answered = False

                                st.session_state.quiz_selected_answer = None

                                st.session_state.quiz_round = 1

                                st.session_state.quiz_history_saved = False

                                st.rerun()

                            except Exception as e:

                                st.error(
                                    f"Could not generate quiz: {e}"
                                )

                # Display quiz

                if st.session_state.quiz_questions:

                    questions = (
                        st.session_state.quiz_questions
                    )

                    current_index = (
                        st.session_state.quiz_current_index
                    )

                    current_question = questions[
                        current_index
                    ]

                    st.progress(
                        (current_index + 1) / 5
                    )

                    st.markdown(
                        f"### Question {current_index + 1} of 5"
                    )

                    st.write(
                        f"**{current_question['question']}**"
                    )

                    options = (
                        current_question["options"]
                    )

                    selected = st.radio(
                        "Choose your answer:",
                        [
                            f"A. {options['A']}",
                            f"B. {options['B']}",
                            f"C. {options['C']}",
                            f"D. {options['D']}"
                        ],
                        key=f"quiz_option_{current_index}"
                    )

                    selected_letter = selected[0]

                    # Submit

                    if not st.session_state.quiz_answered:

                        if st.button(
                            "✅ Submit Answer",
                            use_container_width=True
                        ):

                            st.session_state.quiz_selected_answer = (
                                selected_letter
                            )

                            if (
                                selected_letter
                                ==
                                current_question[
                                    "correct_answer"
                                ]
                            ):

                                st.session_state.quiz_score += 1

                            st.session_state.quiz_answered = True

                            st.rerun()

                    # Result

                    if st.session_state.quiz_answered:

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
                                "🎉 Correct answer!"
                            )

                        else:

                            st.error(
                                "❌ Incorrect answer."
                            )

                            st.info(
                                f"Correct answer: "
                                f"{correct_answer}. "
                                f"{options[correct_answer]}"
                            )

                        st.write(
                            "**Explanation:**"
                        )

                        st.write(
                            current_question[
                                "explanation"
                            ]
                        )

                        # Next question

                        if current_index < 4:

                            if st.button(
                                "➡️ Next Question",
                                use_container_width=True
                            ):

                                st.session_state.quiz_current_index += 1

                                st.session_state.quiz_answered = False

                                st.session_state.quiz_selected_answer = None

                                st.rerun()

                        # Quiz complete

                        else:

                            percentage = (
                                st.session_state.quiz_score
                                / 5
                            ) * 100

                            st.markdown(
                                f"""
                                <div class="score-card">
                                    <h2>🏆 Quiz Complete!</h2>
                                    <h3>
                                        Score:
                                        {st.session_state.quiz_score}/5
                                    </h3>
                                    <h3>
                                        Percentage:
                                        {percentage:.0f}%
                                    </h3>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            if not st.session_state.quiz_history_saved:

                                for q in questions:

                                    st.session_state.quiz_previous_questions.append(
                                        q["question"]
                                    )

                                add_history(
                                    "🎯 Quiz",
                                    f"Quiz Round "
                                    f"{st.session_state.quiz_round}",
                                    (
                                        f"Score: "
                                        f"{st.session_state.quiz_score}/5 "
                                        f"({percentage:.0f}%)"
                                    )
                                )

                                st.session_state.quiz_history_saved = True

                            if st.button(
                                "🔄 Next 5 Questions",
                                use_container_width=True
                            ):

                                with st.spinner(
                                    "Generating new questions..."
                                ):

                                    try:

                                        new_questions = generate_quiz(
                                            full_text,
                                            st.session_state.quiz_previous_questions
                                        )

                                        st.session_state.quiz_questions = (
                                            new_questions
                                        )

                                        st.session_state.quiz_current_index = 0

                                        st.session_state.quiz_score = 0

                                        st.session_state.quiz_answered = False

                                        st.session_state.quiz_selected_answer = None

                                        st.session_state.quiz_round += 1

                                        st.session_state.quiz_history_saved = False

                                        st.rerun()

                                    except Exception as e:

                                        st.error(
                                            f"Could not generate quiz: {e}"
                                        )


            # =================================================
            # DOCUMENT EXPLORER
            # =================================================

            with explorer_tab:

                st.subheader(
                    "📖 Document Explorer"
                )

                st.write(
                    "Explore the chunks created from your PDF."
                )

                st.info(
                    f"This document contains "
                    f"{len(chunks)} chunks."
                )

                selected_chunk = st.number_input(
                    "Select chunk number",
                    min_value=1,
                    max_value=len(chunks),
                    value=1
                )

                chunk_index = (
                    selected_chunk - 1
                )

                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>
                            Chunk {selected_chunk}
                        </strong>
                        <br><br>
                        {chunks[chunk_index]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:

            st.error(
                f"Could not process the PDF: {e}"
            )


# =========================================================
# STUDY WITHOUT PDF
# =========================================================

with topic_mode:

    st.header(
        "📚 Study Without PDF"
    )

    st.write(
        "Don't have notes or a PDF? "
        "Enter your subject and topic, then choose what you want to learn."
    )

    st.warning(
        "For specific language or literature lessons, "
        "your textbook or PDF is recommended because the exact context matters."
    )

    col1, col2 = st.columns(2)

    with col1:

        subject = st.text_input(
            "Subject",
            placeholder="Example: DBMS"
        )

    with col2:

        topic = st.text_input(
            "Topic / Lesson",
            placeholder="Example: File Organization"
        )

    study_type = st.selectbox(
        "What do you want to study?",
        [
            "Simple Explanation",
            "Exam Preparation",
            "Important Points",
            "10-Mark Answer",
            "Quick Revision"
        ]
    )

    if st.button(
        "📚 Start Studying",
        use_container_width=True
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

                lesson = generate_topic_lesson(
                    subject,
                    topic,
                    study_type
                )

            st.write(
                f"**Subject:** {subject}"
            )

            st.write(
                f"**Topic:** {topic}"
            )

            st.write(
                f"**Study mode:** {study_type}"
            )

            st.divider()

            st.markdown(
                lesson
            )

            add_history(
                "📚 Topic Study",
                topic,
                lesson
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 StudyMate RAG • "
    "Learn from your notes. Understand better. Revise smarter."
)