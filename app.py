import heapq
import re
from collections import Counter

import docx
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
<<<<<<< Updated upstream
=======
import heapq
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
>>>>>>> Stashed changes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud


def generate_research_feedback(results):
    feedback = []
    score = results.get("scores", {}).get("Composite", 0)
    semantic = results.get("semantic_strength", 0)
    citations = results.get("citation_analysis", {}).get("total_citations", 0)
    if score < 70:
        feedback.append("Overall structure needs improvement.")
    if semantic < 60:
        feedback.append("Conceptual depth is limited. Add more technical explanations.")
    if citations < 5:
        feedback.append("Add more scholarly references.")
    if "methodology" not in results.get("sections", {}):
        feedback.append("Methodology section is weak or missing.")
    if not feedback:
        feedback.append("Paper demonstrates strong academic quality.")
    return feedback
<<<<<<< Updated upstream


def detect_research_gaps_advanced(sections):
    gap_keywords = [
        "however",
        "future work",
        "limitation",
        "challenge",
        "lack",
        "needs improvement",
        "not addressed",
        "further research",
    ]
    relevant_sections = ["conclusion", "discussion", "future work"]
    gaps = []
    for sec_name, content in sections.items():
        if sec_name.lower() in relevant_sections:
            sentences = content.split(".")
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in gap_keywords):
                    gaps.append(sentence.strip())
    return gaps[:5]


def detect_contribution(text):
    keywords = ["proposed", "novel", "new method", "we introduce", "our contribution"]
    sentences = text.split(".")
    contributions = []
    for sentence in sentences:
        if any(k in sentence.lower() for k in keywords):
            contributions.append(sentence.strip())
    return contributions[:5]


def internal_redundancy_check(text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return 0
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(paragraphs)
    similarity_matrix = cosine_similarity(vectors)
    return round(similarity_matrix.mean() * 100, 2)


=======
>>>>>>> Stashed changes
def recommend_journal(domain):
    journal_map = {
        "Engineering": "IEEE Transactions",
        "AI": "Elsevier Artificial Intelligence Journal",
        "Medical": "Springer Medical Informatics",
        "Legal": "Journal of Legal Analytics",
    }
    return journal_map.get(domain, "Scopus Indexed Multidisciplinary Journal")


st.set_page_config(page_title="PaperIQ", layout="wide")
st.markdown(
    """
<style>
    .summary-box {
        background-color: #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4a4a4a;}
</style>
""",
    unsafe_allow_html=True,
)
if "signed_up" not in st.session_state:
    st.session_state.signed_up = False
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = {}
if not st.session_state.signed_up:
    st.title("📝 PaperIQ Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["Student", "Researcher", "Faculty"])
    if st.button("Sign Up", type="primary", key="signup_btn"):
        if username and password:
            st.session_state.user_data = {"username": username, "role": role}
            st.session_state.signed_up = True
            st.success("Signup Successful! Redirecting...")
            st.rerun()
            for key in list(st.session_state.keys()):
                if key not in ["authenticated", "username"]:
                    del st.session_state[key]
        else:
            st.error("Please fill all fields.")
    st.stop()


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])


def clean_text(text):
    return re.sub(r"\n+", "\n", text).strip()


def extract_sections(text):
    lines = text.split("\n")
    sections = {}
    current_header = "Preamble"
    current_content = []
    common_headers = [
        "ABSTRACT",
        "INTRODUCTION",
        "LITERATURE REVIEW",
        "METHODOLOGY",
        "RESULTS",
        "DISCUSSION",
        "CONCLUSION",
        "REFERENCES",
        "Abstract",
        "Introduction",
        "Methodology",
        "Conclusion",
    ]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_header = False
        if re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", line) and len(line) < 60:
            is_header = True
        elif line in common_headers or (
            line.isupper() and len(line) < 40 and len(line) > 3
        ):
            is_header = True
        if is_header:
            if current_content:
                sections[current_header] = " ".join(current_content)
            current_header = line
            current_content = []
        else:
            current_content.append(line)
    if current_content:
        sections[current_header] = " ".join(current_content)
    return sections


def summarize_text(text, num_sentences=3):
    if not text:
        return "No content to summarize."
    blob = TextBlob(text)
    sentences = blob.sentences
    if len(sentences) <= num_sentences:
        return text
    word_frequencies = {}
    stop_words = set(
        [
            "the",
            "is",
            "in",
            "and",
            "to",
            "of",
            "a",
            "for",
            "on",
            "with",
            "as",
            "by",
            "at",
            "this",
            "that",
            "it",
            "from",
            "an",
            "be",
            "are",
            "was",
        ]
    )
    for word in blob.words:
        word = word.lower()
        if word not in stop_words and word.isalpha():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
    if not word_frequencies:
        return text
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_scores = {}
    for sent in sentences:
        for word in sent.words:
            word = word.lower()
            if word in word_frequencies:
                sentence_scores[sent] = (
                    sentence_scores.get(sent, 0) + word_frequencies[word]
                )
    top_sentences = heapq.nlargest(
        num_sentences, sentence_scores, key=sentence_scores.get
    )
    return " ".join([str(s) for s in top_sentences])


def get_important_sentences(text, num_sentences=3):
    sentences = re.split(r"(?<=[.!?]) +", text)
    if len(sentences) <= num_sentences:
        return sentences
    word_freq = {}
    words = re.findall(r"\w+", text.lower())
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    sentence_scores = {}
    for sentence in sentences:
        for word in re.findall(r"\w+", sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = (
                    sentence_scores.get(sentence, 0) + word_freq[word]
                )
    important = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return important


def count_syllables(word):
    word = word.lower()
    vowels = "aeiou"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def calculate_readability(text):
    sentences = re.split(r"[.!?]+", text)
    words = text.split()
    total_words = len(words)
    total_sentences = len(sentences)
    syllables = sum(count_syllables(word) for word in words)
    if total_sentences == 0 or total_words == 0:
        return 0
    flesch_score = (
        206.835
        - (1.015 * (total_words / total_sentences))
        - (84.6 * (syllables / total_words))
    )
    return round(flesch_score, 2)


def get_grade(score):
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "E"


def analyze_full_document(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    words = blob.words
    word_count = len(words)
    sentence_count = len(sentences)
    if sentence_count == 0:
        return None
    avg_sentence_len = np.mean([len(s.words) for s in sentences])
    avg_word_len = np.mean([len(w) for w in words])
    sentiment = blob.sentiment.polarity
    language_score = min(
        100, (avg_sentence_len * 1.5) + (avg_word_len * 5) + (50 + sentiment * 20)
    )
    transitions = [
        "however",
        "therefore",
        "thus",
        "consequently",
        "furthermore",
        "meanwhile",
    ]
    transition_count = sum(text.lower().count(t) for t in transitions)
    coherence_score = min(100, (transition_count * 4) + (sentence_count * 0.1) + 40)
    reasoning_keywords = [
        "because",
        "since",
        "implies",
        "due to",
        "as a result",
        "evidence",
    ]
    reasoning_count = sum(text.lower().count(k) for k in reasoning_keywords)
    reasoning_score = min(100, (reasoning_count * 6) + 30)
    complex_words = [w for w in words if len(w) > 6]
    lexical_score = (
        min(100, (len(complex_words) / word_count) * 300) if word_count else 0
    )
    readability_score = calculate_readability(text)
    final_score = (
        language_score * 0.3
        + coherence_score * 0.2
        + reasoning_score * 0.2
        + lexical_score * 0.15
        + readability_score * 0.15
    )
    stats = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "avg_word_len": round(avg_word_len, 2),
        "vocab_diversity": round(len(set([w.lower() for w in words])) / word_count, 2)
        if word_count
        else 0,
        "complex_word_ratio": round(len(complex_words) / word_count, 2)
        if word_count
        else 0,
    }
    sections_data = extract_sections(text)
    return {
        "scores": {
            "Language": round(language_score, 2),
            "Coherence": round(coherence_score, 2),
            "Reasoning": round(reasoning_score, 2),
            "Sophistication": round(lexical_score, 2),
            "Readability": round(readability_score, 2),
            "Composite": round(final_score, 2),
        },
        "stats": stats,
        "sentiment": round(sentiment, 2),
        "blob": blob,
        "sections": sections_data,
    }


def generate_full_report(res, filename):
    scores = res["scores"]
    stats = res["stats"]
    keywords = res.get("keywords", [])
    domain = res.get("domain", "Unknown")
    publisher = res.get("publisher", "Not detected")
    sections = res.get("sections", {})
    report = f"""
PAPER ANALYSIS REPORT
========================
File Name: {filename}
Publisher: {publisher}
Domain: {domain}
----------------------------------------
DOCUMENT STATISTICS
----------------------------------------
Word Count: {stats["word_count"]}
Sentence Count: {stats["sentence_count"]}
----------------------------------------
SCORES
----------------------------------------
Language: {scores["Language"]}
Coherence: {scores["Coherence"]}
Reasoning: {scores["Reasoning"]}
Sophistication: {scores["Sophistication"]}
Readability: {scores["Readability"]}
Composite: {scores["Composite"]}
----------------------------------------
KEYWORDS
----------------------------------------
{", ".join(keywords)}
----------------------------------------
SECTION SUMMARIES
----------------------------------------
"""
    for title, summary in sections.items():
        report += f"\n\n{title}\n{'-' * len(title)}\n{summary}\n"
    report += "\n\n----------------------------------------\nEND OF REPORT\n----------------------------------------"
    return report


def clean_for_pdf(text):
    if not text:
        return ""
    return text.encode("latin-1", "ignore").decode("latin-1")


def generate_pdf_report(res, filename):
<<<<<<< Updated upstream
    categories = ["Language", "Coherence", "Reasoning", "Sophistication", "Readability"]
    values = [res["scores"][c] for c in categories]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values, text=values, textposition="auto"))
    fig.update_layout(height=400)
    import uuid

    chart_path = f"chart_{uuid.uuid4().hex}.png"
    fig.write_image(chart_path, format="png", scale=1)
=======
>>>>>>> Stashed changes
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PaperIQ Analysis Report", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"File Name: {filename}")
    pdf.multi_cell(0, 8, clean_for_pdf(f"Title: {res.get('title')}"))
    pdf.multi_cell(0, 8, f"Domain: {res.get('domain')}")
    pdf.multi_cell(0, 8, f"Publisher: {res.get('publisher')}")
    pdf.ln(5)
    pdf.cell(0, 10, "Scores:", ln=True)
    for k, v in res["scores"].items():
        pdf.multi_cell(0, 8, f"{k}: {v}")
    pdf.ln(5)
    pdf.ln(10)
    pdf.cell(0, 10, "Section Summaries:", ln=True)
    for title, summary in res["sections"].items():
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, clean_for_pdf(title))
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, clean_for_pdf(summary))
        pdf.ln(3)
    pdf_output = pdf.output(dest="S")
    return bytes(pdf_output, "latin-1")


def extract_keywords_and_domain(text, top_n=15):
    blob = TextBlob(text)
    stop_words = set(
        [
            "the",
            "is",
            "in",
            "and",
            "to",
            "of",
            "a",
            "for",
            "on",
            "with",
            "as",
            "by",
            "at",
            "this",
            "that",
            "it",
            "from",
            "an",
            "be",
            "are",
            "was",
        ]
    )
    word_freq = {}
    for word in blob.words:
        word = word.lower()
        if word.isalpha() and word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [w[0] for w in sorted_words[:top_n]]
    domains = {
        "Computer Science": ["algorithm", "data", "model", "network", "ai", "learning"],
        "Healthcare": ["patient", "medical", "treatment", "disease", "clinical"],
        "Finance": ["market", "investment", "economic", "revenue", "financial"],
        "Education": ["learning", "students", "teaching", "curriculum"],
        "Engineering": ["system", "design", "performance", "analysis"],
    }
    detected_domain = "General"
    for domain, terms in domains.items():
        for term in terms:
            if term in text.lower():
                detected_domain = domain
                break
    return keywords, detected_domain


def reviewer_comments(score):
    if score > 80:
        return "Strong paper. Minor revisions recommended."
    elif score > 60:
        return "Good work, but clarity and structure need improvement."
    else:
        return "Major revisions required before acceptance."


def semantic_answer(question, full_text, sections):
    question = question.lower()
    section_keywords = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "method": "Methodology",
        "methods": "Methodology",
        "methodology": "Methodology",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
    }
    for key, section_name in section_keywords.items():
        if key in question:
            for title, content in sections.items():
                if section_name.lower() in title.lower():
                    return content[:1000]
    sentences = re.split(r"(?<=[.!?]) +", full_text)
    vectorizer = TfidfVectorizer().fit_transform(sentences + [question])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    best_match_index = np.argmax(cosine_sim)
    return sentences[best_match_index]


STOPWORDS = set(
    [
        "the",
        "is",
        "in",
        "and",
        "to",
        "of",
        "for",
        "on",
        "with",
        "a",
        "an",
        "by",
        "this",
        "that",
        "it",
        "as",
        "are",
        "was",
    ]
)


def detect_repetition(text, threshold=5):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    word_counts = Counter(words)
    repeated = {
        word: count for word, count in word_counts.items() if count >= threshold
    }
    return repeated


def generate_structured_suggestions(text):
    suggestions = []
    sentences = re.split(r"[.!?]", text)
    long_sentences = [s for s in sentences if len(s.split()) > 30]
    if long_sentences:
        suggestions.append(
            "• Consider breaking long sentences into shorter ones for clarity."
        )
    repeated_words = detect_repetition(text)
    if repeated_words:
        suggestions.append(
            "• Reduce repetition of words like: "
            + ", ".join(list(repeated_words.keys())[:5])
        )
    if " was " in text or " were " in text:
        suggestions.append("• Review passive voice usage for stronger academic tone.")
    if not suggestions:
        suggestions.append(
            "• Writing structure looks good. Minor refinements may improve clarity."
        )
    return suggestions


def detect_publisher(text):
    publishers = {
        "IEEE": ["ieee", "ieeexplore"],
        "Springer": ["springer"],
        "Elsevier": ["elsevier", "sciencedirect"],
        "Wiley": ["wiley"],
        "Taylor & Francis": ["taylor", "francis"],
        "ACM": ["acm", "association for computing machinery"],
        "Nature": ["nature publishing"],
        "MDPI": ["mdpi"],
        "Oxford University Press": ["oxford university press"],
    }
    text_lower = text.lower()
    for publisher, keywords in publishers.items():
        for word in keywords:
            if word in text_lower:
                return publisher
    return "Not Detected"


def section_wise_sentiment(sections):
    sentiment_results = {}
    blob = TextBlob(full_text)
    overall_sentiment = blob.sentiment.polarity
    return sentiment_results


def novelty_score(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0
    return round((len(set(words)) / len(words)) * 100, 2)


def analyze_citations(text):
    citation_patterns = [r"\(\d{4}\)", r"\[\d+\]", r"\(\w+ et al\., \d{4}\)"]
    citation_count = 0
    for pattern in citation_patterns:
        citation_count += len(re.findall(pattern, text))
    total_words = len(text.split())
    citation_density = round(citation_count / total_words, 4) if total_words > 0 else 0
    impact_score = min(100, citation_count * 2)
    years = re.findall(r"(19\d{2}|20\d{2})", text)
    if years:
        avg_year = sum(map(int, years)) / len(years)
    else:
        avg_year = 0
    return {
        "total_citations": citation_count,
        "citation_density": citation_density,
        "impact_score": impact_score,
        "average_year": round(avg_year, 1),
    }


def calculate_semantic_strength(text):
    blob = TextBlob(text)
    nouns = [word.lower() for word, tag in blob.tags if tag.startswith("NN")]
    unique_nouns = len(set(nouns))
    total_nouns = len(nouns)
    if total_nouns == 0:
        return 0
    return round(min(100, (unique_nouns / total_nouns) * 150), 2)


def extract_topic_focus(text):
    keywords, _ = extract_keywords_and_domain(text, top_n=15)
    return keywords


st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(135deg, #f4f6fb, #eef1f7);
    font-family: 'Segoe UI', sans-serif;}
h1 {
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6a11cb, #2575fc);
    color: white;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stText,
section[data-testid="stSidebar"] label {
    color: white !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
    transition: 0.3s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);}
[data-testid="stFileUploader"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);}
[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);}
button[data-baseweb="tab"] {
    font-weight: 600;}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #6a11cb !important;
    border-bottom: 3px solid #6a11cb !important;}
.summary-box {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 5px solid #6a11cb;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
    transition: 0.3s ease;}
.summary-box:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 22px rgba(0,0,0,0.1);}
.keyword-tag {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 20px;
    display: inline-block;
    font-size: 13px;}
div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #6a11cb !important;
    border-radius: 10px !important;
    padding: 8px !important;}
div[data-testid="stTextInput"] input:focus {
    border: 2px solid #2575fc !important;
    box-shadow: 0px 0px 8px rgba(106,17,203,0.4);}
.stAlert {
    border-radius: 10px;}
header[data-testid="stHeader"] {
    background: transparent;}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;}
.block-container {
    padding-top: 1rem !important;}
[data-testid="stToolbar"] {
    display: none;}
hr {
    border: none;
    height: 1px;
    background: #e0e0e0;
    margin: 20px 0;}
</style>
""",
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("User Dashboard")
    st.write(f"👤 {st.session_state.user_data['username']}")
    st.write(f"🎓 Role: {st.session_state.user_data['role']}")
    st.markdown("---")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = {}
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = []
st.markdown("### 📤 Upload Research Paper")
uploaded_files = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    if st.button("Analyze Document", type="primary", key="analyze_btn"):
        with st.spinner("Analyzing document..."):
            username = st.session_state.user_data["username"]
            if username not in st.session_state.analysis_history:
                st.session_state.analysis_history[username] = []
            st.session_state.comparison_results = []
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(file)
                    cleaned_text = clean_text(text)
                    results = analyze_full_document(cleaned_text)
                    if results:
                        keywords, domain = extract_keywords_and_domain(cleaned_text)
                        publisher = detect_publisher(cleaned_text)
                        results["keywords"] = keywords
                        results["domain"] = domain
                        results["publisher"] = publisher
                        results["full_text"] = cleaned_text
<<<<<<< Updated upstream
                        results["research_gaps"] = detect_research_gaps_advanced(
                            results["sections"]
                        )
=======
>>>>>>> Stashed changes
                        results["citation_analysis"] = analyze_citations(cleaned_text)
                        results["semantic_strength"] = calculate_semantic_strength(
                            cleaned_text
                        )
                        results["topic_focus"] = extract_topic_focus(cleaned_text)
                        results["ai_feedback"] = generate_research_feedback(results)
<<<<<<< Updated upstream
                        results["recommended_journal"] = recommend_journal(
                            results["domain"]
                        )
                        results["contributions"] = detect_contribution(
                            results["full_text"]
                        )
                        results["redundancy_score"] = internal_redundancy_check(
                            results["full_text"]
                        )
=======
                        results["recommended_journal"] = recommend_journal(results["domain"])
>>>>>>> Stashed changes
                        results["novelty_score"] = novelty_score(cleaned_text)
                        st.session_state.comparison_results.append(results)
                        st.session_state["results"] = results
                        st.session_state["filename"] = file.name
                        pdf_bytes = generate_pdf_report(results, file.name)
                        st.session_state.analysis_history[username].append(
                            {
                                "filename": file.name,
                                "domain": domain,
                                "score": results["scores"]["Composite"],
                                "pdf": pdf_bytes,
                            }
                        )
                else:
                    st.error(f"Unable to analyze file: {file.name}")
if "results" in st.session_state:
    res = st.session_state["results"]
    scores = res["scores"]
    stats = res["stats"]
    st.markdown("## 📈 Analysis Results")
    grade = get_grade(scores["Composite"])
    col_main1, col_main2 = st.columns([2, 1])
    with col_main1:
        st.metric("Composite Score", f"{scores['Composite']}/100")
    with col_main2:
        st.metric("Overall Grade", grade)
        st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Language", f"{scores['Language']}/100")
    col2.metric("Coherence", f"{scores['Coherence']}/100")
    col3.metric("Reasoning", f"{scores['Reasoning']}/100")
    col4.metric("Sophistication", f"{scores['Sophistication']}/100")
    if st.button("Generate PDF Report"):
        pdf_report = generate_pdf_report(res, st.session_state["filename"])
        st.download_button(
            "Download Full Report",
            data=pdf_report,
            file_name="PaperIQ_Report.pdf",
            mime="application/pdf",
        )
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "📊 Visualizations",
            "📑 Section Summaries",
            "📄 Metadata",
            "💡 Suggestions",
            "❤️ Sentiment",
            "📌 Keywords & Domain",
            "Chatbot",
            "📂 Analysis History",
            "🧠 Advanced Insights",
            "📊 Paper Comparison",
        ]
    )
    with tab1:
        st.subheader("📊 Performance Breakdown")
        categories = [
            "Language",
            "Coherence",
            "Reasoning",
            "Sophistication",
            "Readability",
        ]
        values = [scores[c] for c in categories]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=categories, y=values, text=values, textposition="auto"))
        fig.update_layout(
            title="Score Distribution (Out of 100)",
            yaxis=dict(title="Score", range=[0, 100]),
            xaxis=dict(title="Metrics"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("📘 What These Scores Mean")
        st.markdown("""
        **Language** → Sentence quality and structure
        **Coherence** → Logical flow and transitions
        **Reasoning** → Use of arguments and evidence
        **Sophistication** → Vocabulary complexity
        **Readability** → Ease of reading
        """)
    with tab2:
        st.subheader("📑 Smart Section Summaries")
        summary_length = st.radio(
            "Select Summary Length", ["Short", "Medium", "Long"], horizontal=True
        )
        if summary_length == "Short":
            num_sentences = 3
            highlight_sentences = 1
        elif summary_length == "Medium":
            num_sentences = 6
            highlight_sentences = 3
        else:
            num_sentences = 8
            highlight_sentences = 4
        for title, content in res["sections"].items():
            summary = summarize_text(content, num_sentences)
            summary_sentences = re.split(r"(?<=[.!?]) +", summary)
            important_sentences = get_important_sentences(summary, highlight_sentences)
            highlighted_summary = summary
            for sent in important_sentences:
                highlighted_summary = highlighted_summary.replace(
                    sent, f"<mark>{sent}</mark>"
                )
            st.markdown(
                f"""
                <div class="summary-box">
                <b>{title}</b><br><br>
                {highlighted_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
    with tab3:
        st.write(f"**Publisher:** {res['publisher']}")
        st.write(f"**File Name:** {st.session_state['filename']}")
        st.write(f"**Words:** {stats['word_count']}")
        st.write(f"**Sentences:** {stats['sentence_count']}")
    with tab4:
        st.subheader("💡 Writing Suggestions")
        suggestions = generate_structured_suggestions(res["full_text"])
        suggestion_html = "<br>".join(suggestions)
        for point in res["ai_feedback"]:
            st.write("•", point)
        st.success(f"📚 Recommended Journal: {res['recommended_journal']}")
<<<<<<< Updated upstream
        st.subheader("🧬 Key Contributions")
        for c in res["contributions"]:
            st.write("•", c)
        st.info(f"📎 Internal Redundancy Score: {res['redundancy_score']}%")
        st.markdown(
            f"""
=======
        
        st.markdown(f"""
>>>>>>> Stashed changes
        <div class="summary-box">
            {suggestion_html}
        </div>
        """,
            unsafe_allow_html=True,
        )
    with tab5:
        st.subheader("📊 Overall Paper Sentiment")
        sentiment_score = res["sentiment"]
        if sentiment_score > 0:
            label = "Positive"
        elif sentiment_score < 0:
            label = "Negative"
        else:
            label = "Neutral"
        st.metric("Sentiment Score", sentiment_score)
        st.success(f"Overall Tone: {label}")
        st.subheader("Reviewer Simulation")
        st.write(reviewer_comments(res["scores"]["Composite"]))
    with tab6:
<<<<<<< Updated upstream
        st.subheader("📌 Extracted Keywords")
        keyword_html = " ".join(
            [
                f"<span style='background-color:#d9d9d9; padding:6px 10px; margin:5px; border-radius:15px; display:inline-block;'>{kw}</span>"
                for kw in res["keywords"]
            ]
        )
        st.markdown(keyword_html, unsafe_allow_html=True)

        st.subheader("☁️ Keyword Word Cloud")
        if res.get("keywords"):
            keyword_text = " ".join(res["keywords"])
            wc = WordCloud(
                width=1200, height=500, background_color="white", colormap="viridis"
            ).generate(keyword_text)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("No keywords available to generate word cloud.")

        st.markdown("---")
        st.subheader("📚 Detected Domain")
        st.success(res["domain"])
=======
        st.subheader("📚 Detected Domain")
        st.success(res['domain'])
        keywords = res['keywords']
        text_for_cloud = " ".join(keywords)
        wc = WordCloud(width=1000, height=400, background_color="white").generate(text_for_cloud)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
>>>>>>> Stashed changes
    with tab7:
        st.subheader("🤖 Ask About This Paper")
        user_question = st.text_input(
            "Ask a question about the paper", key="paper_chatbot_input"
        )
        if user_question:
            answer = semantic_answer(user_question, res["full_text"], res["sections"])
            st.markdown(
                f"""
            <div class="summary-box">
            <b>Answer:</b> {answer}
            </div>
            """,
                unsafe_allow_html=True,
            )
    with tab8:
        st.subheader("📂 Your Analysis History")
        username = st.session_state.user_data["username"]
        user_history = st.session_state.analysis_history.get(username, [])
        if not user_history:
            st.info("No previous analyses found.")
        else:
            for i, item in enumerate(user_history):
                st.write(f"📄 {item['filename']}")
                st.write(f"**Domain:** {item['domain']}")
                st.write(f"**Composite Score:** {item['score']}")
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "⬇ Re-download Report",
                        data=item["pdf"],
                        file_name=f"{item['filename']}_Report.pdf",
                        mime="application/pdf",
                        key=f"download_{i}",
                    )
                with col2:
                    if st.button("🗑 Delete", key=f"delete_{i}"):
                        st.session_state.analysis_history[username].pop(i)
                        st.rerun()
                st.markdown("---")
    with tab9:
        st.subheader("🧠 Advanced AI Insights")
        citation_data = res.get("citation_analysis", {})
        st.markdown("### 📚 Citation Analysis")
        st.metric("Total Citations Found", citation_data.get("total_citations", 0))
        st.metric("Impact Score", citation_data.get("impact_score", 0))
        st.metric("Average Citation Year", citation_data.get("average_year", 0))
        st.markdown("---")
        st.markdown("### 🧠 Semantic Depth Score")
        st.metric("Semantic Strength", f"{res.get('semantic_strength', 0)}/100")
        st.metric("Citation Density", citation_data.get("citation_density", 0))
        st.markdown("---")
        st.metric("Novelty Score", f"{res.get('novelty_score', 0)}%")
    with tab10:
        st.subheader("📊 Advanced Paper Comparison")
        if (
            "comparison_results" not in st.session_state
            or len(st.session_state.comparison_results) < 2
        ):
            st.warning("Please upload at least 2 papers for comparison.")
        else:
            import matplotlib.pyplot as plt
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            papers = st.session_state.comparison_results
            max_score = max(
                paper.get("scores", {}).get("Composite", 0) for paper in papers
            )
            comparison_data = []
            for i, paper in enumerate(papers):
                citation_data = paper.get("citation_analysis", {})
                total_citations = (
                    citation_data.get("total_citations")
                    or citation_data.get("citation_count")
                    or citation_data.get("count")
                    or citation_data.get("citations")
                    or 0
                )
                composite_score = paper.get("scores", {}).get("Composite", 0)
                comparison_data.append(
                    {
                        "Paper": f"Paper {i + 1}",
                        "Domain": paper.get("domain", "N/A"),
                        "Composite Score": composite_score,
                        "Semantic Strength": paper.get("semantic_strength", 0),
                        "Total Citations": total_citations,
                        "Best Paper": "⭐" if composite_score == max_score else "",
                    }
                )
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            best_paper = max(
                papers, key=lambda x: x.get("scores", {}).get("Composite", 0)
            )
            try:
                if len(papers) >= 2:
                    texts = [paper.get("full_text", "") for paper in papers]
                    vectorizer = TfidfVectorizer(stop_words="english")
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    st.subheader("📊 Similarity Matrix (%)")
                    import pandas as pd

                    similarity_df = pd.DataFrame(
                        similarity_matrix * 100,
                        columns=[f"Paper {i + 1}" for i in range(len(papers))],
                        index=[f"Paper {i + 1}" for i in range(len(papers))],
                    )
                    st.dataframe(similarity_df.round(2))
            except:
                st.warning("Similarity calculation unavailable.")
            scores = [paper.get("scores", {}).get("Composite", 0) for paper in papers]
            labels = [f"Paper {i + 1}" for i in range(len(scores))]
            plt.figure()
            plt.bar(labels, scores)
            plt.xlabel("Papers")
            plt.ylabel("Composite Score")
            st.pyplot(plt)
