import time
import os
import re
import io
import json
import sqlite3

import pymupdf
import pytesseract
from PIL import Image
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin, unquote

import requests
from bs4 import BeautifulSoup

import streamlit as st
from pypdf import PdfReader
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Online question-paper search
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StudyMate-RAG",
    page_icon="📚",
    layout="wide"
)

DB_PATH = "studymate_history.db"
PAPER_FOLDER = Path("question_papers")

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]
LANGUAGES = [
    "English", "Kannada", "Hindi", "Tamil", "Telugu",
    "Marathi", "Urdu", "Odia", "Punjabi", "Gujarati",
    "Bengali", "Malayalam"
]

# OCR configuration (Tesseract must be installed on this computer).
TESSERACT_EXE = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

OCR_LANGUAGES = {
    "English": "eng",
    "Kannada": "kan",
    "Hindi": "hin",
    "Tamil": "tam",
    "Telugu": "tel",
    "Marathi": "mar",
    "Urdu": "urd",
    "Odia": "ori",
    "Punjabi": "pan",
    "Gujarati": "guj",
    "Bengali": "ben",
    "Malayalam": "mal",
}

EDUCATION_LEVELS = [
    "School",
    "PUC / Higher Secondary",
    "Degree / University",
    "Competitive Exam",
    "Other"
]

PAPER_TYPES = [
    "Previous Year Question Paper",
    "Sample Paper",
    "Model Paper",
    "Practice Paper",
    "Other"
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        color: #234b83;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #65748b;
        margin-bottom: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 12px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 9px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# GEMINI CONNECTION
# =========================================================

def get_api_key():
    key = os.getenv("GEMINI_API_KEY", "")

    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""

    return key


def ask_ai(prompt):
    api_key = get_api_key()

    if not api_key:
        return (
            "ERROR: Gemini API key is missing. Add GEMINI_API_KEY "
            "to your environment variables or Streamlit secrets."
        )

    models = [MODEL_NAME] + [
        model for model in FALLBACK_MODELS
        if model != MODEL_NAME
    ]

    
    errors = []

    for model_name in models:
        for attempt in range(2):
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options={"timeout": 180000}
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                result = getattr(response, "text", None)

                if result and result.strip():
                    return result.strip()

                errors.append(
                    f"{model_name}: Empty response"
                )
                break

            except Exception as exc:
                message = str(exc)

                errors.append(
                    f"{model_name}, attempt {attempt + 1}: "
                    f"{message[:200]}"
                )

                temporary = any(
                    term in message.upper()
                    for term in (
                        "503",
                        "504",
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                        "429",
                        "RESOURCE_EXHAUSTED"
                    )
                )

                if temporary and attempt == 0:
                    time.sleep(2)
                    continue

                break

    return (
        "ERROR: Gemini could not respond after retries. "
        "Check model availability, API quota, and internet connection.\n\n"
        + "\n".join(errors)
    )



def is_error(text):
    return (
        isinstance(text, str)
        and text.strip().startswith("ERROR:")
    )


def language_instruction():
    language = st.session_state.language

    return (
        f"Answer in {language}. Use simple, student-friendly language. "
        "Keep technical names accurate. Follow the selected answer "
        "language even if the question is written in another language."
    )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                mode TEXT,
                activity TEXT,
                question TEXT,
                answer TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                topic TEXT,
                score INTEGER,
                total INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS question_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                education_level TEXT NOT NULL,
                board_university TEXT NOT NULL,
                class_course TEXT NOT NULL,
                exam_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                year INTEGER NOT NULL,
                paper_type TEXT NOT NULL,
                source_url TEXT,
                file_path TEXT,
                created_at TEXT
            )
        """)


def save_history(mode, activity, question, answer):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO study_history
                (created_at, mode, activity, question, answer)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                mode,
                activity,
                question,
                answer
            ))
    except sqlite3.Error:
        pass


def save_quiz_result(topic, score, total):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO quiz_results
                (created_at, topic, score, total)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                topic,
                score,
                total
            ))
    except sqlite3.Error:
        pass


def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT created_at, mode, activity, question, answer
            FROM study_history
            ORDER BY id DESC
            LIMIT 100
        """).fetchall()


def load_quiz_results():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT created_at, topic, score, total
            FROM quiz_results
            ORDER BY id DESC
            LIMIT 50
        """).fetchall()


def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM study_history")


def clear_quiz_results():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM quiz_results")


# =========================================================
# QUESTION PAPER LIBRARY DATABASE FUNCTIONS
# =========================================================

def load_question_papers():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("""
            SELECT id, title, education_level, board_university,
                   class_course, exam_name, subject, year,
                   paper_type, source_url, file_path, created_at
            FROM question_papers
            ORDER BY year DESC, title COLLATE NOCASE
        """).fetchall()


def add_question_paper(
    title,
    education_level,
    board_university,
    class_course,
    exam_name,
    subject,
    year,
    paper_type,
    source_url="",
    file_path=""
):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO question_papers (
                title, education_level, board_university,
                class_course, exam_name, subject, year,
                paper_type, source_url, file_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title.strip(),
            education_level,
            board_university.strip(),
            class_course.strip(),
            exam_name.strip(),
            subject.strip(),
            int(year),
            paper_type,
            source_url.strip(),
            file_path,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))


def delete_question_paper(paper_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT file_path FROM question_papers WHERE id = ?",
            (paper_id,)
        ).fetchone()

        conn.execute(
            "DELETE FROM question_papers WHERE id = ?",
            (paper_id,)
        )

    if row and row[0]:
        try:
            file_path = Path(row[0]).resolve()
            allowed_folder = PAPER_FOLDER.resolve()

            if (
                file_path.is_relative_to(allowed_folder)
                and file_path.is_file()
            ):
                file_path.unlink()
        except (OSError, ValueError):
            pass


def get_safe_paper_filename(original_name):
    filename = os.path.basename(original_name)
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{timestamp}_{filename}"


def is_valid_http_url(url):
    if not url.strip():
        return True

    try:
        parsed = urlparse(url.strip())

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.netloc)
        )
    except ValueError:
        return False


def init_app():
    init_db()
    PAPER_FOLDER.mkdir(parents=True, exist_ok=True)


init_app()


# =========================================================
# ONLINE QUESTION PAPER SEARCH
# =========================================================

def search_online_papers(query, max_results=10):
    """
    Search the web for question-paper pages.
    Results are search results, not verified or downloaded PDFs.
    """

    if DDGS is None:
        return [], (
            "The online search package is not installed. "
            "Install it with: pip install ddgs"
        )

    try:
        results = []

        with DDGS() as search_client:
            raw_results = search_client.text(
                query,
                max_results=max_results
            )

            for item in raw_results:
                title = str(item.get("title", "")).strip()
                url = str(item.get("href", "")).strip()
                snippet = str(item.get("body", "")).strip()

                if not title or not is_valid_http_url(url):
                    continue

                if not url:
                    continue

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

        return results, ""

    except Exception as exc:
        return [], (
            "Online search failed. Check your internet connection "
            f"and try again. Details: {str(exc)[:200]}"
        )


def make_paper_search_query(
    subject,
    course,
    university,
    year,
    paper_type
):
    parts = []

    if subject.strip():
        parts.append(subject.strip())

    if course.strip():
        parts.append(course.strip())

    if university.strip():
        parts.append(university.strip())

    if year != "Any year":
        parts.append(year)

    if paper_type != "Any paper type":
        parts.append(paper_type)

    parts.extend([
        "question paper",
        "previous year exam paper PDF"
    ])

    return " ".join(parts)


def save_online_result(result, subject, course, university, year):
    title = result["title"]
    url = result["url"]

    safe_year = (
        int(year)
        if str(year).isdigit()
        else datetime.now().year
    )

    level = (
        "Competitive Exam"
        if any(word in course.lower()
               for word in ["jee", "neet", "upsc", "competitive"])
        else "Degree / University"
    )

    add_question_paper(
        title=title,
        education_level=level,
        board_university=university.strip() or "Not specified",
        class_course=course.strip() or "Not specified",
        exam_name="Question Paper",
        subject=subject.strip() or "Not specified",
        year=safe_year,
        paper_type="Previous Year Question Paper",
        source_url=url,
        file_path=""
    )


def rank_paper_results(results, subject, course, university, year):
    """Rank search results by relevance; this does not verify authenticity."""
    import re
    from urllib.parse import urlparse

    def words(value):
        return set(re.findall(r"[a-z0-9]+", str(value).lower()))

    subject_words = words(subject)
    course_words = words(course)
    university_words = words(university)

    ranked = []

    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")
        text = f"{title} {snippet} {urlparse(url).netloc}".lower()
        text_words = words(text)

        score = 0

        # Subject is essential: reward exact subject terms.
        if subject_words:
            matched = subject_words & text_words
            score += 5 * len(matched) / len(subject_words)

            # Common abbreviations and full subject names.
            aliases = {
                "cn": ["computer networks", "computer network"],
                "dbms": ["database management systems", "database management"],
                "os": ["operating systems", "operating system"],
            }
            for alias in aliases.get(subject.lower().strip(), []):
                if alias in text:
                    score += 5

        # Course and university matches.
        if course_words:
            score += 4 * len(course_words & text_words) / len(course_words)

        if university_words:
            score += 4 * len(university_words & text_words) / len(university_words)

        # Reward a requested year only when it appears in the result text.
        if str(year).isdigit() and str(year) in text:
            score += 3

        # Prefer likely paper/PDF results.
        if "question paper" in text or "question papers" in text:
            score += 2
        if ".pdf" in url.lower() or "pdf" in title.lower():
            score += 1

        ranked.append((score, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for _, result in ranked]


# =========================================================
# DOWNLOAD AND VERIFY ONLINE QUESTION-PAPER PDFs
# =========================================================

def _is_pdf_response(response):
    content_type = (response.headers.get("content-type") or "").lower()
    return (
        "application/pdf" in content_type
        or response.content[:5] == b"%PDF-"
    )


def _download_candidate_pdf(url, timeout=18):
    """Return validated PDF bytes or (None, reason); never trust URL suffix alone."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StudyMateRAG/1.0)"},
            allow_redirects=True,
        )
        response.raise_for_status()
        if not _is_pdf_response(response):
            return None, "The URL did not return PDF content."
        ok, reason = extract_paper_pdf(response.content)
        if not ok:
            return None, reason
        return response.content, response.url
    except Exception as exc:
        return None, str(exc)[:180]


def discover_pdf_links(page_url, search_title=""):
    """Discover PDF links on a result page and prioritize likely relevant titles."""
    parsed = urlparse(page_url)
    if parsed.scheme not in ("http", "https"):
        return []
    if parsed.path.lower().endswith(".pdf"):
        return [page_url]
    try:
        response = requests.get(
            page_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StudyMateRAG/1.0)"},
            allow_redirects=True,
        )
        response.raise_for_status()
        if _is_pdf_response(response):
            return [response.url]
        content_type = (response.headers.get("content-type") or "").lower()
        if "html" not in content_type and "text/" not in content_type:
            return []
        soup = BeautifulSoup(response.text[:2_000_000], "html.parser")
        links = []
        title_terms = set(re.findall(r"[a-z0-9]+", (search_title or "").lower()))
        for a in soup.find_all("a", href=True):
            href = urljoin(response.url, a.get("href", "").strip())
            if urlparse(href).scheme not in ("http", "https"):
                continue
            label = " ".join([a.get_text(" ", strip=True), href]).lower()
            if ".pdf" in urlparse(href).path.lower() or "pdf" in label:
                score = sum(term in label for term in title_terms if len(term) > 2)
                links.append((score, href))
        links.sort(key=lambda item: item[0], reverse=True)
        output, seen = [], set()
        for _, link in links:
            if link not in seen:
                output.append(link)
                seen.add(link)
            if len(output) >= 12:
                break
        return output
    except Exception:
        return []


def save_downloaded_paper_pdf(pdf_bytes, title, level, board, course, exam, subject, year, paper_type, source_url):
    """Validate and store an online PDF inside the paper library folder and DB."""
    valid, message = extract_paper_pdf(pdf_bytes)
    if not valid:
        raise ValueError(message)
    PAPER_FOLDER.mkdir(parents=True, exist_ok=True)
    safe_name = get_safe_paper_filename((title or "question_paper") + ".pdf")
    destination = PAPER_FOLDER / safe_name
    destination.write_bytes(pdf_bytes)
    try:
        add_question_paper(
            title=title or "Downloaded question paper",
            education_level=level or "Degree / University",
            board_university=board or "Not specified",
            class_course=course or "Not specified",
            exam_name=exam or "Question Paper",
            subject=subject or "Not specified",
            year=int(year),
            paper_type=paper_type or "Previous Year Question Paper",
            source_url=source_url or "",
            file_path=str(destination),
        )
    except Exception:
             destination.unlink(missing_ok=True)
             raise
    return str(destination)


# =========================================================
# REPEATED QUESTION ANALYSIS
# =========================================================

def normalize_question(text):
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"^\s*(?:q(?:uestion)?\s*)?\d+[.)\]:-]?\s*", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_questions_from_paper(text):
    """Extract likely question lines from extracted PDF text, with sentence fallback."""
    candidates = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < 18 or len(line) > 700:
            continue
        numbered = re.match(r"^(?:Q(?:uestion)?\s*)?\d{1,3}\s*[.)\]:-]\s*(.+)$", line, re.I)
        if numbered:
            line = numbered.group(1).strip()
        elif not ("?" in line or re.match(r"^(?:explain|define|describe|discuss|what|why|how|compare|differentiate|write|list|state|derive|prove|calculate|illustrate|elaborate|short note|give an account)\b", line, re.I)):
            continue
        line = re.sub(r"\s*_{3,}\s*", " ", line).strip(" -•\t")
        norm = normalize_question(line)
        if len(norm.split()) >= 4 and norm not in {normalize_question(x) for x in candidates}:
            candidates.append(line)
    return candidates


def analyze_repeated_questions(papers, similarity_threshold=0.78):
    """Return exact repeats and conceptually similar question clusters across saved PDFs."""
    records = []
    for paper in papers:
        paper_id, title, level, board, course, exam, subject, year, p_type, url, file_path, created_at = paper
        if not file_path:
            continue
        path = Path(file_path)
        try:
            if not path.is_file() or not path.resolve().is_relative_to(PAPER_FOLDER.resolve()):
                continue
            reader = PdfReader(str(path))
            pages_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception:
            continue
        for question in extract_questions_from_paper(pages_text):
            norm = normalize_question(question)
            records.append({"question": question, "norm": norm, "year": int(year), "title": title, "subject": subject, "paper_id": paper_id})
    if not records:
        return {"records": [], "exact": [], "similar": [], "error": "No readable questions found in saved local PDFs. Upload papers as PDFs to your library first; scanned image-only PDFs may need OCR."}

    exact_map = {}
    for record in records:
        if record["norm"]:
            exact_map.setdefault(record["norm"], []).append(record)
    exact = [items for items in exact_map.values() if len({x["paper_id"] for x in items}) > 1]
    similar = []
    # Greedy clusters keep the UI understandable and avoid listing every pair.
    texts = [r["norm"] for r in records]
    if len(texts) >= 2:
        try:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
            matrix = vectorizer.fit_transform(texts)
            sims = cosine_similarity(matrix)
            used = set()
            for i in range(len(records)):
                if i in used:
                    continue
                cluster = [i]
                for j in range(i + 1, len(records)):
                    if j in used or records[i]["norm"] == records[j]["norm"]:
                        continue
                    if sims[i, j] >= similarity_threshold:
                        cluster.append(j)
                distinct_papers = {records[k]["paper_id"] for k in cluster}
                if len(cluster) > 1 and len(distinct_papers) > 1:
                    similar.append([records[k] for k in cluster])
                    used.update(cluster)
        except ValueError:
            pass
    return {"records": records, "exact": exact, "similar": similar, "error": ""}


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "language": "English",
    "study_mode": "Study With PDF",
    "pages": [],
    "chunks": [],
    "filename": "",
    "pdf_signature": "",
    "last_answer": "",
    "last_question": "",
    "last_activity": "",
    "summary": "",
    "quiz_questions": [],
    "quiz_topic": "",
    "quiz_submitted": False,
    "quiz_score": 0,
    "exam_answer": "",
    "evaluation": "",
    "exam_question": "",
    "document_search": "",
    "online_paper_results": [],
    "online_paper_search_error": "",
    "online_paper_search_query": "",
    "repeated_question_analysis": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CLEAR RESULTS WHEN MODE CHANGES
# =========================================================

def clear_mode_results():
    st.session_state.last_answer = ""
    st.session_state.last_question = ""
    st.session_state.last_activity = ""
    st.session_state.summary = ""
    st.session_state.quiz_questions = []
    st.session_state.quiz_topic = ""
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.exam_answer = ""
    st.session_state.evaluation = ""
    st.session_state.exam_question = ""


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf(uploaded_file, ocr_languages=None):
    """Extract selectable PDF text and OCR scanned pages when needed."""
    if ocr_languages is None:
        ocr_languages = ["eng"]

    allowed_codes = set(OCR_LANGUAGES.values())
    selected_codes = [code for code in ocr_languages if code in allowed_codes]
    if not selected_codes:
        selected_codes = ["eng"]
    language_string = "+".join(selected_codes)

    try:
        pdf_bytes = uploaded_file.getvalue()
        pages = []

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for index, page in enumerate(document):
                try:
                    text = (page.get_text("text") or "").strip()
                except Exception:
                    text = ""

                ocr_used = False
                # OCR only when normal extraction found no selectable text.
                if not text:
                    try:
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2),
                            alpha=False
                        )
                        image = Image.frombytes(
                            "RGB", (pix.width, pix.height), pix.samples
                        )
                        text = pytesseract.image_to_string(
                            image,
                            lang=language_string,
                            config="--oem 1 --psm 3"
                        ).strip()
                        ocr_used = True
                    except Exception as ocr_exc:
                        print(f"OCR failed on PDF page {index + 1}: {ocr_exc}")
                        text = ""

                pages.append({
                    "page": index + 1,
                    "text": text,
                    "ocr_used": ocr_used,
                })

        return pages

    except Exception as exc:
        st.error(f"Could not read the PDF: {exc}")
        return []


def extract_paper_pdf(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        if len(reader.pages) == 0:
            return False, "This PDF has no pages."

        return True, ""

    except Exception:
        return False, "The uploaded file is not a valid PDF."


# =========================================================
# CHUNKING AND RETRIEVAL
# =========================================================

def create_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()

            if piece:
                chunks.append({
                    "page": page["page"],
                    "text": piece
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def retrieve(question, chunks, top_k=5):
    if not question.strip() or not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            matrix
        ).flatten()

        ranked = scores.argsort()[::-1][:top_k]
        results = []

        for index in ranked:
            if scores[index] <= 0:
                continue

            results.append({
                "page": chunks[index]["page"],
                "text": chunks[index]["text"],
                "score": float(scores[index])
            })

        return results

    except Exception:
        return []


def get_document_context(question):
    matches = retrieve(
        question,
        st.session_state.chunks
    )

    if not matches:
        return "", []

    context_parts = []
    page_numbers = []

    for item in matches:
        context_parts.append(
            f"[PDF page {item['page']}]\n{item['text']}"
        )
        page_numbers.append(item["page"])

    return (
        "\n\n".join(context_parts),
        sorted(set(page_numbers))
    )


def get_all_document_text(limit=25000):
    parts = []

    for page in st.session_state.pages:
        if page["text"]:
            parts.append(
                f"[PDF page {page['page']}]\n{page['text']}"
            )

    return "\n\n".join(parts)[:limit]


def has_document():
    return bool(
        st.session_state.pages
        and st.session_state.chunks
    )


def document_required_message():
    st.info(
        "Upload a PDF and select Study With PDF to use this feature."
    )


# =========================================================
# ANSWER GENERATION
# =========================================================

def answer_question(question):
    mode = st.session_state.study_mode
    language = language_instruction()

    if mode == "Study Without PDF":
        prompt = f"""
You are StudyMate-RAG, a helpful study assistant.

{language}

The student is studying WITHOUT a PDF. Answer using general
educational knowledge. Do not refer to a previously uploaded PDF.

QUESTION:
{question}

Give a clear answer with a suitable definition, explanation,
and examples when useful. If the question is unclear, explain
what you understood.
"""
        return ask_ai(prompt), []

    if not has_document():
        return (
            "Please upload a readable PDF before asking questions "
            "in Study With PDF mode.",
            []
        )

    context, pages = get_document_context(question)

    if not context:
        return (
            "I could not find readable text relevant to your question "
            "in this PDF. Try words that appear in your notes or check "
            "whether the PDF contains selectable text.",
            []
        )

    prompt = f"""
You are StudyMate-RAG, an accurate PDF-based study assistant.

{language}

Answer the student's question using the PDF context below.

STRICT RULES:
1. Check whether the supplied context answers the question.
2. Do not invent facts or claim unsupported information comes from PDF.
3. If the PDF does not explain the topic, clearly say:
   "This topic is not covered in the retrieved PDF text."
4. You may add a separate section called
   "General explanation (outside the PDF)" if useful.
   Clearly distinguish it from the notes.
5. Cite page numbers only when supported by the supplied context.
6. Use simple, exam-friendly language.

PDF CONTEXT:
{context}

STUDENT QUESTION:
{question}
"""

    return ask_ai(prompt), pages


# =========================================================
# SUMMARY
# =========================================================

def generate_summary(topic, pdf_mode=False):
    language = language_instruction()

    if pdf_mode:
        if not has_document():
            return "ERROR: Please upload a PDF first."

        source = get_all_document_text()

        prompt = f"""
Create a clear student-friendly summary of the supplied PDF.

{language}

Rules:
- Use only the supplied PDF.
- Do not invent details.
- Use headings and bullet points.
- Preserve important names, definitions, dates and concepts.
- Say when information is missing or unreadable.

PDF TEXT:
{source}
"""
    else:
        prompt = f"""
Create a useful educational summary about this topic.

{language}

TOPIC:
{topic}

Use a definition, important points, examples and a short conclusion
where appropriate. Do not refer to an uploaded PDF.
"""

    return ask_ai(prompt)


# =========================================================
# QUIZ GENERATION
# =========================================================

def generate_quiz(topic, pdf_mode=False, number=5):
    language = language_instruction()

    if pdf_mode:
        if not has_document():
            return "ERROR: Please upload a PDF first."

        source = get_all_document_text()

        prompt = f"""
Create {number} multiple-choice questions based ONLY on this PDF.

{language}

Return valid JSON only:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0,
      "explanation": "Explanation supported by the PDF"
    }}
  ]
}}

The answer must be an integer from 0 to 3.
Do not include markdown fences or text outside JSON.
Do not invent facts beyond the PDF.

PDF TEXT:
{source}
"""
    else:
        prompt = f"""
Create {number} multiple-choice questions about this topic.

{language}

TOPIC:
{topic}

Return valid JSON only:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer": 0,
      "explanation": "Explanation"
    }}
  ]
}}

The answer must be an integer from 0 to 3.
Do not include markdown fences or text outside JSON.
Do not refer to a previously uploaded PDF.
"""

    response = ask_ai(prompt)

    if is_error(response):
        return response

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response.strip(),
        flags=re.IGNORECASE
    )

    try:
        data = json.loads(cleaned)
        questions = data.get("questions", [])
        valid_questions = []

        for item in questions:
            if not isinstance(item, dict):
                continue

            options = item.get("options", [])
            answer = item.get("answer")

            if (
                isinstance(item.get("question"), str)
                and isinstance(options, list)
                and len(options) == 4
                and all(isinstance(option, str) for option in options)
                and isinstance(answer, int)
                and not isinstance(answer, bool)
                and 0 <= answer < 4
            ):
                valid_questions.append(item)

        if not valid_questions:
            return "ERROR: The AI did not return valid quiz questions."

        return valid_questions

    except (json.JSONDecodeError, AttributeError, TypeError):
        return "ERROR: Could not read the generated quiz. Please try again."


# =========================================================
# EXAM ANSWERS AND EVALUATION
# =========================================================

def generate_exam_answer(question, marks, pdf_mode):
    language = language_instruction()

    if pdf_mode:
        if not has_document():
            return "ERROR: Please upload a PDF first."

        context, pages = get_document_context(question)

        if not context:
            return (
                "I could not find relevant information in the PDF. "
                "Try a question using terms from your notes."
            )

        prompt = f"""
Prepare an exam-style answer worth {marks} marks.

{language}

Use the supplied PDF context as the source.
Do not invent facts. If the notes do not contain the answer,
say so clearly. Include page references when supported.

PDF CONTEXT:
{context}

QUESTION:
{question}
"""
    else:
        prompt = f"""
Prepare an exam-style answer worth {marks} marks.

{language}

Use general educational knowledge. Do not refer to a PDF.
Use a suitable heading, definition and numbered points.
Include examples or a conclusion when appropriate.

QUESTION:
{question}
"""

    return ask_ai(prompt)


def evaluate_exam_answer(question, student_answer, pdf_mode):
    language = language_instruction()

    if pdf_mode:
        if not has_document():
            return "ERROR: Please upload a PDF first."

        context, _ = get_document_context(question)

        if not context:
            return (
                "I could not find the relevant topic in the PDF, "
                "so I cannot reliably evaluate this answer against "
                "your notes."
            )

        source_instruction = f"""
Evaluate against the supplied notes.
Do not treat unsupported information as if it came from the PDF.

PDF CONTEXT:
{context}
"""
    else:
        source_instruction = """
Evaluate using general educational knowledge.
Do not refer to a previously uploaded PDF.
"""

    prompt = f"""
You are a supportive exam-answer evaluator.

{language}

{source_instruction}

QUESTION:
{question}

STUDENT'S ANSWER:
{student_answer}

Give:
1. What is correct
2. What is missing or needs correction
3. A suggested improved answer
4. A score out of 10, with a short reason

Be fair and constructive.
"""

    return ask_ai(prompt)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("📚 StudyMate")
    st.caption("Your multilingual AI-powered study assistant")

    st.subheader("🌐 Language Settings")

    st.selectbox(
        "Answer language",
        LANGUAGES,
        key="language"
    )

    st.divider()
    st.subheader("📊 Study History")

    try:
        st.metric("Study activities", len(load_history()))
        st.metric("Quiz attempts", len(load_quiz_results()))
    except Exception:
        st.caption("History information is unavailable.")

    st.divider()
    st.subheader("ℹ️ About")

    st.caption(
        "StudyMate-RAG helps you study with PDF notes or learn "
        "general topics without a PDF."
    )

    st.markdown("**Workflow**")
    st.caption("PDF → Text → Chunks → Retrieval → Gemini → Answer")


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown(
    '<div class="main-title">📚 StudyMate-RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your multilingual AI-powered study assistant'
    '</div>',
    unsafe_allow_html=True
)

st.success("🎓 Learn smarter. Revise faster.")
st.divider()
st.subheader("Choose Your Study Mode")

st.radio(
    "Select how you want to study:",
    ["Study With PDF", "Study Without PDF"],
    key="study_mode",
    horizontal=True,
    on_change=clear_mode_results
)


# =========================================================
# PDF UPLOADER
# =========================================================

if st.session_state.study_mode == "Study With PDF":
    st.subheader("📄 Upload Your Study Notes")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="main_page_pdf",
        help="Upload a PDF. Scanned pages can be read using OCR; select the languages below."
    )

    if uploaded_file is not None:
        ocr_language_names = st.multiselect(
            "OCR languages for scanned pages",
            options=list(OCR_LANGUAGES.keys()),
            default=["English", "Kannada"],
            help="Choose the languages printed in scanned/image-based pages.",
            key="ocr_languages_selected",
        )
        ocr_language_codes = [
            OCR_LANGUAGES[name] for name in ocr_language_names
        ]
        pdf_bytes = uploaded_file.getvalue()
        signature = (
            f"{uploaded_file.name}-"
            f"{uploaded_file.size}-"
            f"{hash(pdf_bytes)}-"
            f"{','.join(ocr_language_codes)}"
        )

        if signature != st.session_state.pdf_signature:
            with st.spinner(
                "Reading your PDF and preparing study notes..."
            ):
                pages = extract_pdf(
                    uploaded_file,
                    ocr_languages=ocr_language_codes,
                )
                chunks = create_chunks(pages)

                st.session_state.pages = pages
                st.session_state.chunks = chunks
                st.session_state.filename = uploaded_file.name
                st.session_state.pdf_signature = signature

                st.session_state.last_answer = ""
                st.session_state.last_question = ""
                st.session_state.summary = ""
                st.session_state.quiz_questions = []
                st.session_state.quiz_submitted = False
                st.session_state.exam_answer = ""
                st.session_state.evaluation = ""

        if has_document():
            readable_pages = sum(
                1 for page in st.session_state.pages
                if page["text"]
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("PDF pages", len(st.session_state.pages))
            col2.metric("Readable pages", readable_pages)
            col3.metric("Text chunks", len(st.session_state.chunks))

            st.success(f"Loaded: {st.session_state.filename}")

            ocr_pages = [
                page["page"] for page in st.session_state.pages
                if page.get("ocr_used", False)
            ]
            if ocr_pages:
                st.info(
                    "OCR was used on page(s): "
                    + ", ".join(map(str, ocr_pages))
                )

            if readable_pages < len(st.session_state.pages):
                st.warning(
                    "Some pages have no extractable text. "
                    "Scanned pages may require OCR."
                )
        else:
            st.error(
                "No readable text was extracted. Please check the PDF."
            )

    elif not has_document():
        st.info("Upload a PDF to start studying from your notes.")
    else:
        st.info(
            f"Using previously loaded notes: {st.session_state.filename}"
        )

else:
    st.info(
        "📖 Study Without PDF is active. Answers in this mode use "
        "general educational knowledge, not your previously uploaded PDF."
    )


# =========================================================
# DYNAMIC TABS
# =========================================================

if st.session_state.study_mode == "Study With PDF":
    tab_names = [
        "📖 Study",
        "📝 Summary",
        "❓ Quiz",
        "🔎 Document Explorer",
        "🎯 Exam Preparation",
        "📚 Previous Question Papers",
        "🔁 Repeated Questions",
        "📊 History"
    ]
else:
    tab_names = [
        "📖 Study",
        "📝 Summary",
        "❓ Quiz",
        "🎯 Exam Preparation",
        "📚 Previous Question Papers",
        "🔁 Repeated Questions",
        "📊 History"
    ]

tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, tabs))


# =========================================================
# STUDY TAB
# =========================================================

with tab_map["📖 Study"]:
    st.subheader("Ask a Study Question")

    with st.form("study_question_form"):
        question = st.text_area(
            "Enter your question",
            placeholder=(
                "Example: Explain Python in simple words"
                if st.session_state.study_mode == "Study Without PDF"
                else "Example: Explain the main points from my notes"
            ),
            height=100,
            key="study_question_input"
        )

        ask_button = st.form_submit_button(
            "✨ Get Answer",
            use_container_width=True
        )

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        elif (
            st.session_state.study_mode == "Study With PDF"
            and not has_document()
        ):
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Preparing your answer..."):
                answer, pages = answer_question(question.strip())

            st.session_state.last_question = question.strip()
            st.session_state.last_answer = answer
            st.session_state.last_activity = "Study"

            if not is_error(answer):
                save_history(
                    st.session_state.study_mode,
                    "Study",
                    question.strip(),
                    answer
                )

    if st.session_state.last_answer:
        st.divider()
        st.subheader("Answer")
        st.markdown(st.session_state.last_answer)

        if (
            st.session_state.study_mode == "Study With PDF"
            and st.session_state.last_question
            and has_document()
        ):
            _, page_refs = get_document_context(
                st.session_state.last_question
            )

            if page_refs:
                st.caption(
                    "Relevant retrieved PDF pages: "
                    + ", ".join(map(str, page_refs))
                )


# =========================================================
# SUMMARY TAB
# =========================================================

with tab_map["📝 Summary"]:
    st.subheader("📝 Generate a Summary")

    pdf_mode = st.session_state.study_mode == "Study With PDF"

    if pdf_mode:
        if not has_document():
            document_required_message()
        else:
            st.caption(f"Summary source: {st.session_state.filename}")

            if st.button(
                "Generate PDF Summary",
                key="generate_pdf_summary",
                use_container_width=True
            ):
                with st.spinner("Summarizing your notes..."):
                    result = generate_summary("", pdf_mode=True)

                st.session_state.summary = result

                if not is_error(result):
                    save_history(
                        st.session_state.study_mode,
                        "Summary",
                        st.session_state.filename,
                        result
                    )

            if st.session_state.summary:
                st.markdown(st.session_state.summary)

    else:
        topic = st.text_input(
            "Enter a subject or topic",
            placeholder="Example: Python basics",
            key="general_summary_topic"
        )

        if st.button(
            "Generate Topic Summary",
            key="generate_general_summary",
            use_container_width=True
        ):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Creating your summary..."):
                    result = generate_summary(topic.strip())

                st.session_state.summary = result

                if not is_error(result):
                    save_history(
                        st.session_state.study_mode,
                        "Summary",
                        topic.strip(),
                        result
                    )

        if st.session_state.summary:
            st.markdown(st.session_state.summary)


# =========================================================
# QUIZ TAB
# =========================================================

with tab_map["❓ Quiz"]:
    st.subheader("❓ Test Your Knowledge")

    pdf_mode = st.session_state.study_mode == "Study With PDF"

    if pdf_mode and not has_document():
        document_required_message()
    else:
        if pdf_mode:
            topic = st.session_state.filename
            st.caption("Questions will be based on your uploaded PDF.")
        else:
            topic = st.text_input(
                "Enter a subject or topic for the quiz",
                placeholder="Example: Computer networks",
                key="general_quiz_topic"
            )

        if st.button(
            "Generate Quiz",
            key="generate_quiz_button",
            use_container_width=True
        ):
            if not topic.strip():
                st.warning("Please enter a topic.")
            else:
                with st.spinner("Generating questions..."):
                    result = generate_quiz(topic, pdf_mode=pdf_mode)

                if isinstance(result, str):
                    if is_error(result):
                        st.error(result)
                    else:
                        st.warning(result)
                else:
                    st.session_state.quiz_questions = result
                    st.session_state.quiz_topic = topic
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0

        questions = st.session_state.quiz_questions

        if questions:
            st.divider()
            st.write(f"**Quiz topic:** {st.session_state.quiz_topic}")

            selected_answers = []

            for index, item in enumerate(questions):
                st.markdown(
                    f"**Q{index + 1}. {item['question']}**"
                )

                options = item["options"]

                selected = st.radio(
                    f"Choose an answer for question {index + 1}",
                    options,
                    index=None,
                    key=f"quiz_answer_{index}",
                    label_visibility="collapsed",
                    disabled=st.session_state.quiz_submitted
                )

                selected_answers.append(
                    options.index(selected)
                    if selected in options else None
                )

            if not st.session_state.quiz_submitted:
                if st.button(
                    "Submit Quiz",
                    key="submit_quiz_button",
                    use_container_width=True
                ):
                    score = sum(
                        1
                        for index, item in enumerate(questions)
                        if selected_answers[index] == item["answer"]
                    )

                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True

                    save_quiz_result(
                        st.session_state.quiz_topic,
                        score,
                        len(questions)
                    )

                    save_history(
                        st.session_state.study_mode,
                        "Quiz",
                        st.session_state.quiz_topic,
                        f"Score: {score}/{len(questions)}"
                    )

                    st.rerun()

            if st.session_state.quiz_submitted:
                score = st.session_state.quiz_score
                total = len(questions)

                st.success(f"Your score: {score}/{total}")

                for index, item in enumerate(questions):
                    correct = item["answer"]
                    chosen = selected_answers[index]

                    if chosen == correct:
                        st.markdown(f"✅ **Q{index + 1}: Correct**")
                    else:
                        st.markdown(f"❌ **Q{index + 1}: Incorrect**")
                        st.write(
                            f"Correct answer: {item['options'][correct]}"
                        )

                    st.caption(item.get("explanation", ""))

                if st.button(
                    "Try Another Quiz",
                    key="reset_quiz_button"
                ):
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = 0
                    st.rerun()


# =========================================================
# DOCUMENT EXPLORER — PDF MODE ONLY
# =========================================================

if "🔎 Document Explorer" in tab_map:
    with tab_map["🔎 Document Explorer"]:
        st.subheader("🔎 Document Explorer")

        if not has_document():
            document_required_message()
        else:
            page_options = [
                page["page"]
                for page in st.session_state.pages
                if page["text"]
            ]

            if page_options:
                selected_page = st.selectbox(
                    "Choose a PDF page",
                    page_options,
                    key="explorer_page"
                )

                page_data = next(
                    (
                        page for page in st.session_state.pages
                        if page["page"] == selected_page
                    ),
                    None
                )

                if page_data:
                    st.markdown(f"**Page {selected_page}**")

                    st.text_area(
                        "Extracted page text",
                        value=page_data["text"],
                        height=350,
                        disabled=True,
                        key=f"page_text_{selected_page}"
                    )

            st.divider()
            st.subheader("Search Within the PDF")

            search_term = st.text_input(
                "Enter a word or phrase",
                key="document_search_input"
            )

            if search_term.strip():
                found = False

                for page in st.session_state.pages:
                    text = page["text"]

                    if search_term.casefold() in text.casefold():
                        found = True
                        st.markdown(
                            f"**Found on page {page['page']}**"
                        )

                        position = text.casefold().find(
                            search_term.casefold()
                        )

                        start = max(0, position - 150)
                        end = min(
                            len(text),
                            position + len(search_term) + 250
                        )

                        st.write(text[start:end])

                if not found:
                    st.info(
                        "No exact match was found in the extracted text."
                    )


# =========================================================
# EXAM PREPARATION TAB
# =========================================================

with tab_map["🎯 Exam Preparation"]:
    st.subheader("🎯 Exam Preparation")

    pdf_mode = st.session_state.study_mode == "Study With PDF"

    if pdf_mode and not has_document():
        document_required_message()
    else:
        exam_question = st.text_area(
            "Enter your exam question",
            placeholder="Example: Explain the types of computer networks",
            height=90,
            key="exam_question_input"
        )

        marks = st.selectbox(
            "Marks",
            [2, 5, 10],
            index=1,
            key="exam_marks"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Generate Exam Answer",
                key="generate_exam_answer_button",
                use_container_width=True
            ):
                if not exam_question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Preparing your exam answer..."):
                        result = generate_exam_answer(
                            exam_question.strip(),
                            marks,
                            pdf_mode
                        )

                    st.session_state.exam_answer = result
                    st.session_state.exam_question = exam_question.strip()
                    st.session_state.evaluation = ""

                    if not is_error(result):
                        save_history(
                            st.session_state.study_mode,
                            "Exam Preparation",
                            exam_question.strip(),
                            result
                        )

        with col2:
            if st.button(
                "Clear Exam Answer",
                key="clear_exam_answer_button",
                use_container_width=True
            ):
                st.session_state.exam_answer = ""
                st.session_state.evaluation = ""
                st.rerun()

        if st.session_state.exam_answer:
            st.divider()
            st.markdown(st.session_state.exam_answer)

        st.divider()
        st.subheader("Evaluate Your Answer")

        student_answer = st.text_area(
            "Write your answer here",
            height=150,
            key="student_exam_answer"
        )

        if st.button(
            "Evaluate My Answer",
            key="evaluate_exam_answer_button",
            use_container_width=True
        ):
            if not exam_question.strip():
                st.warning("Enter the exam question first.")
            elif not student_answer.strip():
                st.warning("Write your answer first.")
            else:
                with st.spinner("Evaluating your answer..."):
                    result = evaluate_exam_answer(
                        exam_question.strip(),
                        student_answer.strip(),
                        pdf_mode
                    )

                st.session_state.evaluation = result

                if not is_error(result):
                    save_history(
                        st.session_state.study_mode,
                        "Answer Evaluation",
                        exam_question.strip(),
                        result
                    )

        if st.session_state.evaluation:
            st.markdown(st.session_state.evaluation)


# =========================================================
# PREVIOUS QUESTION PAPERS LIBRARY
# =========================================================

with tab_map["📚 Previous Question Papers"]:
    st.subheader("📚 Previous Question Paper Library")

    st.write(
        "Search online for question papers from different universities, "
        "boards, courses and competitive exams. Save useful results in "
        "your personal question-paper library."
    )

    st.info(
        "Online search results are links, not verified documents. "
        "Check the website and year before relying on a paper. "
        "Only papers you save or upload are added to your library."
    )

    # -----------------------------------------------------
    # OFFICIAL SOURCE LINKS
    # -----------------------------------------------------

    st.markdown("### 🌐 Official Question Paper Sources")

    official_sources = [
        (
            "CBSE Previous Year Question Papers",
            "https://www.cbse.gov.in/cbsenew/question-paper.html"
        ),
        (
            "CBSE Sample Question Papers",
            "https://cbseacademic.nic.in/sqp_archive.html"
        ),
        (
            "JEE Main Official Website",
            "https://jeemain.nta.nic.in/"
        ),
        (
            "NEET Official Website",
            "https://neet.nta.nic.in/archive/"
        ),
        (
            "UPSC Official Website",
            "https://upsc.gov.in/"
        ),
    ]

    for source_name, source_url in official_sources:
        st.markdown(f"- [{source_name}]({source_url})")

    st.divider()

    # -----------------------------------------------------
    # ONLINE SEARCH FORM
    # -----------------------------------------------------

    st.markdown("### 🔎 Search Question Papers Online")

    current_year = datetime.now().year
    year_options = ["Any year"] + [
        str(year)
        for year in range(current_year, 1999, -1)
    ]

    search_type_options = [
        "Any paper type",
        "Previous Year Question Paper",
        "Sample Paper",
        "Model Paper",
        "Practice Paper"
    ]

    with st.form("online_paper_search_form"):
        online_subject = st.text_input(
            "Subject or exam",
            placeholder="Example: DBMS, Java, Mathematics, NEET"
        )

        online_course = st.text_input(
            "Course / class / semester",
            placeholder="Example: BCA 3rd Semester, Class 12, JEE"
        )

        online_university = st.text_input(
            "University / board / exam authority",
            placeholder="Example: Bengaluru City University, Delhi University"
        )

        search_col1, search_col2 = st.columns(2)

        with search_col1:
            online_year = st.selectbox(
                "Year",
                year_options
            )

        with search_col2:
            online_type = st.selectbox(
                "Paper type",
                search_type_options
            )

        online_search_button = st.form_submit_button(
            "🌐 Search Online",
            use_container_width=True
        )

    
    if online_search_button:
        if not any([
            online_subject.strip(),
            online_course.strip(),
            online_university.strip()
        ]):
            st.warning(
                "Enter at least a subject, course, or university."
            )
            st.session_state.online_paper_results = []
            st.session_state.online_paper_search_query = ""
            st.session_state.online_paper_search_error = ""
        else:
            query = make_paper_search_query(
                online_subject,
                online_course,
                online_university,
                online_year,
                online_type
            )

            st.session_state.online_paper_search_query = query

            with st.spinner("Searching the web for question papers..."):
                results, error = search_online_papers(
                    query,
                    max_results=10
                )

                if not error and results:
                    results = rank_paper_results(
                        results,
                        online_subject,
                        online_course,
                        online_university,
                        online_year
                    )

            st.session_state.online_paper_results = results
            st.session_state.online_paper_search_error = error

    # -----------------------------------------------------
    # DISPLAY SEARCH STATUS AND RESULTS
    # -----------------------------------------------------

    if st.session_state.online_paper_search_error:
        st.error(st.session_state.online_paper_search_error)

    results = st.session_state.online_paper_results

    if results:
        st.markdown("#### Search Results")

        inferred_year = (
            int(online_year)
            if online_year.isdigit()
            else current_year
        )

        inferred_type = (
            online_type
            if online_type != "Any paper type"
            else "Previous Year Question Paper"
        )

        inferred_level = (
            "Competitive Exam"
            if any(
                word in (
                    online_course + " " + online_subject
                ).lower()
                for word in [
                    "jee", "neet", "upsc", "competitive", "gate"
                ]
            )
            else "Degree / University"
        )

        for index, result in enumerate(results):
            with st.container(border=True):
                st.markdown(
                    f"**{index + 1}. {result['title']}**"
                )

                if result.get("snippet"):
                    st.write(result["snippet"])

                st.markdown(
                    f"[🔗 Open source website]({result['url']})"
                )

                save_col1, save_col2 = st.columns(2)

                with save_col1:
                    if st.button(
                        "💾 Save source link",
                        key=f"save_online_paper_{index}",
                        use_container_width=True
                    ):
                        try:
                            with sqlite3.connect(DB_PATH) as conn:
                                existing = conn.execute(
                                    """
                                    SELECT id FROM question_papers
                                    WHERE source_url = ?
                                    LIMIT 1
                                    """,
                                    (result["url"],)
                                ).fetchone()

                            if existing:
                                st.info(
                                    "This source is already in your library."
                                )
                            else:
                                add_question_paper(
                                    title=result["title"],
                                    education_level=inferred_level,
                                    board_university=(
                                        online_university.strip()
                                        or "Not specified"
                                    ),
                                    class_course=(
                                        online_course.strip()
                                        or "Not specified"
                                    ),
                                    exam_name="Question Paper",
                                    subject=(
                                        online_subject.strip()
                                        or "Not specified"
                                    ),
                                    year=inferred_year,
                                    paper_type=inferred_type,
                                    source_url=result["url"],
                                    file_path=""
                                )

                                st.success(
                                    "Source link saved. Use Find & download "
                                    "PDF to store a verified PDF file."
                                )

                        except Exception as exc:
                            st.error(
                                f"Could not save this result: {exc}"
                            )

                with save_col2:
                    if st.button(
                        "⬇️ Find & download PDF",
                        key=f"download_online_paper_{index}",
                        use_container_width=True
                    ):
                        with st.spinner(
                            "Checking the result page for a real PDF..."
                        ):
                            candidates = discover_pdf_links(
                                result["url"],
                                result["title"]
                            )

                            downloaded = None
                            final_source = ""
                            last_error = (
                                "No PDF links were found on this page."
                            )

                            for candidate in candidates:
                                data, outcome = _download_candidate_pdf(
                                    candidate
                                )

                                if data:
                                    downloaded = data
                                    final_source = outcome
                                    break

                                last_error = outcome

                        if not downloaded:
                            st.warning(
                                "No valid downloadable PDF was found. "
                                f"{last_error}"
                            )
                        else:
                            try:
                                with sqlite3.connect(DB_PATH) as conn:
                                    existing = conn.execute(
                                        """
                                        SELECT id FROM question_papers
                                        WHERE source_url = ?
                                        LIMIT 1
                                        """,
                                        (final_source,)
                                    ).fetchone()

                                if existing:
                                    st.info(
                                        "This PDF source is already "
                                        "in your library."
                                    )
                                else:
                                    saved = save_downloaded_paper_pdf(
                                        downloaded,
                                        result["title"],
                                        inferred_level,
                                        online_university.strip()
                                        or "Not specified",
                                        online_course.strip()
                                        or "Not specified",
                                        "Question Paper",
                                        online_subject.strip()
                                        or "Not specified",
                                        inferred_year,
                                        inferred_type,
                                        final_source
                                    )

                                    st.success(
                                        "Verified PDF downloaded and saved: "
                                        f"{Path(saved).name}"
                                    )

                                    st.download_button(
                                        "Download this PDF now",
                                        data=downloaded,
                                        file_name=Path(saved).name,
                                        mime="application/pdf",
                                        key=f"download_verified_pdf_{index}"
                                    )

                            except Exception as exc:
                                st.error(
                                    f"Could not save the PDF: {exc}"
                                )

    elif (
        st.session_state.online_paper_search_query
        and not st.session_state.online_paper_search_error
    ):
        st.info(
            "No results were returned. Try a broader subject or "
            "remove the year or university restriction."
        )

    st.divider()


    # -----------------------------------------------------
    # MANUALLY ADD A PAPER
    # -----------------------------------------------------

    st.markdown("### ➕ Add a Question Paper")

    st.caption(
        "You can manually add paper details and optionally upload a PDF."
    )

    with st.form("add_question_paper_form", clear_on_submit=True):
        paper_title = st.text_input(
            "Paper title *",
            placeholder="Example: BCA 3rd Semester DBMS Question Paper"
        )

        education_level = st.selectbox(
            "Education level *",
            EDUCATION_LEVELS
        )

        board_university = st.text_input(
            "Board / University / Exam Authority *",
            placeholder="Example: Bengaluru City University"
        )

        class_course = st.text_input(
            "Class / Course / Semester *",
            placeholder="Example: BCA 3rd Semester"
        )

        exam_name = st.text_input(
            "Exam name *",
            placeholder="Example: Semester Examination"
        )

        subject = st.text_input(
            "Subject *",
            placeholder="Example: Database Management Systems"
        )

        year = st.number_input(
            "Paper year *",
            min_value=1950,
            max_value=current_year + 1,
            value=current_year,
            step=1
        )

        paper_type = st.selectbox(
            "Paper type *",
            PAPER_TYPES
        )

        source_url = st.text_input(
            "Source URL (optional)",
            placeholder="Leave blank if you don't have a link"
        )

        uploaded_paper = st.file_uploader(
            "Upload question paper PDF (optional)",
            type=["pdf"],
            key="question_paper_upload"
        )

        submit_paper = st.form_submit_button(
            "💾 Save Question Paper",
            use_container_width=True
        )

    if submit_paper:
        required_fields = [
            paper_title.strip(),
            board_university.strip(),
            class_course.strip(),
            exam_name.strip(),
            subject.strip()
        ]

        if not all(required_fields):
            st.error(
                "Please fill in all required fields marked with *."
            )

        elif source_url.strip() and not is_valid_http_url(source_url):
            st.error(
                "The source URL must begin with http:// or https:// "
                "and contain a valid host."
            )

        else:
            saved_path = ""
            validation_passed = True

            try:
                if uploaded_paper is not None:
                    pdf_bytes = uploaded_paper.getvalue()

                    valid_pdf, error_message = extract_paper_pdf(
                        pdf_bytes
                    )

                    if not valid_pdf:
                        st.error(error_message)
                        validation_passed = False
                    else:
                        safe_name = get_safe_paper_filename(
                            uploaded_paper.name
                        )

                        destination = PAPER_FOLDER / safe_name

                        with open(destination, "wb") as pdf_file:
                            pdf_file.write(pdf_bytes)

                        saved_path = str(destination)

                if validation_passed:
                    add_question_paper(
                        title=paper_title,
                        education_level=education_level,
                        board_university=board_university,
                        class_course=class_course,
                        exam_name=exam_name,
                        subject=subject,
                        year=int(year),
                        paper_type=paper_type,
                        source_url=source_url,
                        file_path=saved_path
                    )

                    st.success(
                        f"Saved '{paper_title}' to your question paper library."
                    )

            except Exception as exc:
                if saved_path:
                    try:
                        Path(saved_path).unlink(missing_ok=True)
                    except OSError:
                        pass

                st.error(f"Could not save the question paper: {exc}")

    st.divider()

    # -----------------------------------------------------
    # SEARCH AND FILTER SAVED LIBRARY
    # -----------------------------------------------------

    st.markdown("### 📂 Your Saved Question Papers")

    all_papers = load_question_papers()

    if not all_papers:
        st.info(
            "Your library is empty. Search online and save a result, "
            "or add a paper manually above."
        )

    else:
        st.metric("Saved papers", len(all_papers))

        search_query = st.text_input(
            "Search by title, board, course, exam or subject",
            placeholder="Example: BCA, DBMS, CBSE, NEET",
            key="paper_search_query"
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_level = st.selectbox(
                "Filter by education level",
                ["All"] + EDUCATION_LEVELS,
                key="paper_level_filter"
            )

        with filter_col2:
            available_years = sorted(
                {paper[7] for paper in all_papers},
                reverse=True
            )

            selected_year = st.selectbox(
                "Filter by year",
                ["All"] + available_years,
                key="paper_year_filter"
            )

        filtered_papers = []
        query = search_query.casefold().strip()

        for paper in all_papers:
            (
                paper_id,
                title,
                level,
                board,
                course,
                exam,
                paper_subject,
                paper_year,
                p_type,
                url,
                file_path,
                created_at
            ) = paper

            searchable_text = " ".join([
                str(title),
                str(level),
                str(board),
                str(course),
                str(exam),
                str(paper_subject),
                str(paper_year),
                str(p_type)
            ]).casefold()

            if query and query not in searchable_text:
                continue

            if selected_level != "All" and level != selected_level:
                continue

            if (
                selected_year != "All"
                and paper_year != selected_year
            ):
                continue

            filtered_papers.append(paper)

        st.write(f"**Matching papers: {len(filtered_papers)}**")

        if not filtered_papers:
            st.warning("No papers match these search filters.")

        else:
            for paper in filtered_papers:
                (
                    paper_id,
                    title,
                    level,
                    board,
                    course,
                    exam,
                    paper_subject,
                    paper_year,
                    p_type,
                    url,
                    file_path,
                    created_at
                ) = paper

                with st.expander(f"📄 {title} — {paper_year}"):
                    st.write(f"**Education level:** {level}")
                    st.write(f"**Board / University:** {board}")
                    st.write(f"**Class / Course:** {course}")
                    st.write(f"**Exam:** {exam}")
                    st.write(f"**Subject:** {paper_subject}")
                    st.write(f"**Year:** {paper_year}")
                    st.write(f"**Paper type:** {p_type}")
                    st.caption(f"Added: {created_at}")

                    if url:
                        st.markdown(
                            f"[Open original source]({url})"
                        )

                    if file_path:
                        try:
                            local_path = Path(file_path).resolve()
                            allowed_folder = PAPER_FOLDER.resolve()

                            if (
                                local_path.is_relative_to(allowed_folder)
                                and local_path.is_file()
                            ):
                                with open(local_path, "rb") as pdf_file:
                                    pdf_data = pdf_file.read()

                                st.download_button(
                                    "⬇️ Download PDF",
                                    data=pdf_data,
                                    file_name=local_path.name,
                                    mime="application/pdf",
                                    key=f"download_paper_{paper_id}"
                                )
                            else:
                                st.warning(
                                    "The saved PDF file could not be found. "
                                    "You may need to upload it again."
                                )

                        except OSError:
                            st.warning(
                                "Could not open the saved PDF file."
                            )

                    if st.button(
                        "🗑️ Delete this library entry",
                        key=f"delete_paper_{paper_id}"
                    ):
                        delete_question_paper(paper_id)
                        st.success("Paper entry deleted.")
                        st.rerun()


# =========================================================
# REPEATED QUESTION ANALYSIS TAB
# =========================================================

if "🔁 Repeated Questions" in tab_map:
    with tab_map["🔁 Repeated Questions"]:
        st.subheader("🔁 Repeated-question and topic-frequency analysis")
        st.write("Analyze PDFs saved in your question-paper library. Exact repeats are matched after text normalization; similar questions are estimated using text similarity and should be reviewed by a student.")
        local_papers = [p for p in load_question_papers() if p[10] and Path(p[10]).is_file()]
        st.metric("Saved local PDFs available", len(local_papers))
        threshold = st.slider("Similarity threshold", min_value=0.60, max_value=0.95, value=0.78, step=0.01, help="Lower values find more similar pairs but may include false matches.")
        if st.button("Analyze saved question papers", type="primary", use_container_width=True):
            with st.spinner("Extracting questions and comparing papers..."):
                st.session_state["repeated_question_analysis"] = analyze_repeated_questions(local_papers, threshold)
        analysis = st.session_state.get("repeated_question_analysis")
        if analysis:
            if analysis.get("error"):
                st.info(analysis["error"])
            else:
                st.success(f"Analyzed {len(analysis['records'])} question candidates from {len({r['paper_id'] for r in analysis['records']})} PDF(s).")
                st.markdown("### Exact repeated questions")
                st.caption("These match after case, punctuation, and spacing normalization.")
                if not analysis["exact"]:
                    st.info("No exact repeated questions were detected across different papers.")
                for group in sorted(analysis["exact"], key=lambda g: (-len({x['paper_id'] for x in g}), g[0]['norm'])):
                    years = sorted({x["year"] for x in group})
                    st.markdown(f"**Found in {len({x['paper_id'] for x in group})} papers · Years: {', '.join(map(str, years))}**")
                    st.write(group[0]["question"])
                    st.caption("Sources: " + " | ".join(f"{x['title']} ({x['year']})" for x in group))
                st.markdown("### Similar questions / recurring concepts")
                if not analysis["similar"]:
                    st.info("No similar question clusters were detected at this threshold.")
                for idx, group in enumerate(sorted(analysis["similar"], key=lambda g: (-len({x['paper_id'] for x in g}), g[0]['norm'])), start=1):
                    years = sorted({x["year"] for x in group})
                    with st.expander(f"Concept cluster {idx} · {len({x['paper_id'] for x in group})} papers · Years {', '.join(map(str, years))}"):
                        for item in group:
                            st.markdown(f"- **{item['year']} — {item['title']}**: {item['question']}")
                st.markdown("### Paper coverage")
                counts = {}
                for item in analysis["records"]:
                    counts.setdefault((item["title"], item["year"]), 0)
                    counts[(item["title"], item["year"])] += 1
                for (title, year), count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0][1])):
                    st.write(f"- **{year} — {title}:** {count} extracted question candidates")
        st.caption("Note: text extraction can miss questions in scanned/image-only PDFs. Similarity clusters are estimates, not proof that questions are identical.")


# =========================================================
# HISTORY TAB
# =========================================================

with tab_map["📊 History"]:
    st.subheader("📊 Study History")

    with st.expander("🗑️ Clear Study History"):
        st.warning("This permanently deletes study history.")
        confirm_history = st.checkbox(
            "I confirm clearing study history",
            key="confirm_clear_history"
        )

        if st.button("Clear Study History", key="clear_history_btn"):
            if confirm_history:
                clear_history()
                st.success("Study history cleared.")
                st.rerun()
            else:
                st.warning("Please tick the confirmation box first.")

    history = load_history()

    if history:
        for item in history:
            created_at, mode, activity, question, answer = item

            with st.expander(f"{created_at} · {activity} · {mode}"):
                st.markdown(f"**Question or topic:** {question}")
                st.markdown(answer)
    else:
        st.info("Your study activity will appear here.")

    st.divider()
    st.subheader("Quiz Progress")

    with st.expander("🗑️ Clear Quiz Attempts"):
        st.warning("This permanently deletes quiz results.")
        confirm_quiz = st.checkbox(
            "I confirm clearing quiz attempts",
            key="confirm_clear_quiz"
        )

        if st.button("Clear Quiz Attempts", key="clear_quiz_btn"):
            if confirm_quiz:
                clear_quiz_results()
                st.success("Quiz attempts cleared.")
                st.rerun()
            else:
                st.warning("Please tick the confirmation box first.")

    quiz_results = load_quiz_results()

    if quiz_results:
        for created_at, topic, score, total in quiz_results:
            percentage = score / total * 100 if total else 0
            st.write(
                f"**{topic}** — {score}/{total} "
                f"({percentage:.0f}%) · {created_at}"
            )
    else:
        st.info("Your quiz results will appear here.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "StudyMate-RAG | Learn smarter. Revise faster. "
    "Check important answers against your original notes and textbooks."
)