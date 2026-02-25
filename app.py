
import streamlit as st
import re
import numpy as np
import pdfplumber
import docx
from textblob import TextBlob
import plotly.graph_objects as go
from fpdf import FPDF
import heapq
from collections import Counter
st.set_page_config(page_title="PaperIQ", layout="wide")
st.markdown("""
<style>

    .summary-box {
        background-color: #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #4a4a4a;
    }

</style>
""", unsafe_allow_html=True)

if "signed_up" not in st.session_state:
    st.session_state.signed_up = False

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if not st.session_state.signed_up:

    st.title("📝 PaperIQ Signup")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["Student", "Researcher", "Faculty"])

    if st.button("Sign Up", type="primary", key="signup_btn"):

        if username and password:
            st.session_state.user_data = {
                "username": username,
                "role": role
            }
            st.session_state.signed_up = True
            st.success("Signup Successful! Redirecting...")
            st.rerun()
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
    return re.sub(r'\n+', '\n', text).strip()

def extract_sections(text):
    lines = text.split('\n')
    sections = {}
    current_header = "Preamble"
    current_content = []

    common_headers = [
        "ABSTRACT", "INTRODUCTION", "LITERATURE REVIEW", "METHODOLOGY",
        "RESULTS", "DISCUSSION", "CONCLUSION", "REFERENCES",
        "Abstract", "Introduction", "Methodology", "Conclusion"
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_header = False

        if re.match(r'^\d+(\.\d+)*\s+[A-Za-z]', line) and len(line) < 60:
            is_header = True
        elif line in common_headers or (line.isupper() and len(line) < 40 and len(line) > 3):
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
    stop_words = set([
        'the','is','in','and','to','of','a','for','on','with','as',
        'by','at','this','that','it','from','an','be','are','was'
    ])

    for word in blob.words:
        word = word.lower()
        if word not in stop_words and word.isalpha():
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if not word_frequencies:
        return text

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    sentence_scores = {}
    for sent in sentences:
        for word in sent.words:
            word = word.lower()
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    top_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    return " ".join([str(s) for s in top_sentences])

def section_wise_sentiment(sections):
    sentiment_results = {}

    for title, content in sections.items():
        blob = TextBlob(content)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        sentiment_results[title] = {
            "polarity": round(polarity, 3),
            "label": label
        }

    return sentiment_results
def get_important_sentences(text, num_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= num_sentences:
        return sentences

    word_freq = {}
    words = re.findall(r'\w+', text.lower())

    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = {}
    for sentence in sentences:
        for word in re.findall(r'\w+', sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

    important = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return important
def calculate_readability(text):
    sentences = text.count('.') + text.count('!') + text.count('?')
    words = len(text.split())
    syllables = int(words * 1.5)

    if sentences == 0 or words == 0:
        return 0

    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return max(0, min(100, score))

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

    language_score = min(100, (avg_sentence_len * 1.5) + (avg_word_len * 5) + (50 + sentiment * 20))

    transitions = ["however", "therefore", "thus", "consequently", "furthermore", "meanwhile"]
    transition_count = sum(text.lower().count(t) for t in transitions)
    coherence_score = min(100, (transition_count * 4) + (sentence_count * 0.1) + 40)

    reasoning_keywords = ["because", "since", "implies", "due to", "as a result", "evidence"]
    reasoning_count = sum(text.lower().count(k) for k in reasoning_keywords)
    reasoning_score = min(100, (reasoning_count * 6) + 30)

    complex_words = [w for w in words if len(w) > 6]
    lexical_score = min(100, (len(complex_words) / word_count) * 300) if word_count else 0
    readability_score = calculate_readability(text)

    final_score = (
        language_score * 0.3 +
        coherence_score * 0.2 +
        reasoning_score * 0.2 +
        lexical_score * 0.15 +
        readability_score * 0.15
    )

    stats = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
        "avg_word_len": round(avg_word_len, 2),
        "vocab_diversity": round(len(set(words.lower())) / word_count, 2) if word_count else 0,
        "complex_word_ratio": round(len(complex_words) / word_count, 2) if word_count else 0
    }

    sections_data = extract_sections(text)
    

    return {
        "scores": {
            "Language": round(language_score, 2),
            "Coherence": round(coherence_score, 2),
            "Reasoning": round(reasoning_score, 2),
            "Sophistication": round(lexical_score, 2),
            "Readability": round(readability_score, 2),
            "Composite": round(final_score, 2)
        },
        "stats": stats,
        "sentiment": round(sentiment, 2),
        "blob": blob,
        "sections": sections_data
    }
def detect_title(text):
    lines = text.split("\n")

    for line in lines[:15]:  
        line = line.strip()
        if len(line) > 15 and len(line.split()) > 3:
            return line

    return "Title not detected"

def generate_full_report(res, filename):

    scores = res["scores"]
    stats = res["stats"]
    keywords = res.get("keywords", [])
    domain = res.get("domain", "Unknown")
    publisher = res.get("publisher", "Not detected")
    title = res.get("title", "Unknown")
    sections = res.get("sections", {})

    report = f"""
PAPER ANALYSIS REPORT
========================

File Name: {filename}
Title: {title}
Publisher: {publisher}
Domain: {domain}

----------------------------------------
DOCUMENT STATISTICS
----------------------------------------
Word Count: {stats['word_count']}
Sentence Count: {stats['sentence_count']}

----------------------------------------
SCORES
----------------------------------------
Language: {scores['Language']}
Coherence: {scores['Coherence']}
Reasoning: {scores['Reasoning']}
Sophistication: {scores['Sophistication']}
Readability: {scores['Readability']}
Composite: {scores['Composite']}

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

    categories = ['Language', 'Coherence', 'Reasoning', 'Sophistication', 'Readability']
    values = [res['scores'][c] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=values, text=values, textposition='auto'))
    fig.update_layout(height=400)

    chart_path = "chart.png"
    fig.write_image(chart_path ,format="png", scale=1)

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
    for k, v in res['scores'].items():
        pdf.multi_cell(0, 8, f"{k}: {v}")

    pdf.ln(5)
    pdf.image(chart_path, x=10, w=180)
    pdf.ln(10)

    pdf.cell(0, 10, "Section Summaries:", ln=True)

    for title, summary in res['sections'].items():
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, clean_for_pdf(title))
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, clean_for_pdf(summary))
        pdf.ln(3)

    pdf_output = pdf.output(dest='S')
    return bytes(pdf_output, "latin-1")

def extract_keywords_and_domain(text, top_n=10):
    blob = TextBlob(text)

    stop_words = set([
        'the','is','in','and','to','of','a','for','on','with','as',
        'by','at','this','that','it','from','an','be','are','was'
    ])

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
        "Engineering": ["system", "design", "performance", "analysis"]
    }

    detected_domain = "General"

    for domain, terms in domains.items():
        for term in terms:
            if term in text.lower():
                detected_domain = domain
                break

    return keywords, detected_domain
def paper_chatbot(question, sections, full_res):
    question = question.lower().strip()

    

    if any(word in question for word in ["objective", "aim", "purpose"]):
        for title, content in sections.items():
            if "abstract" in title.lower() or "introduction" in title.lower():
                return summarize_text(content, 3)

    if any(word in question for word in ["method", "methodology", "approach"]):
        for title, content in sections.items():
            if "method" in title.lower():
                return summarize_text(content, 3)

    if "result" in question or "finding" in question:
        for title, content in sections.items():
            if "result" in title.lower():
                return summarize_text(content, 3)

    if "conclusion" in question:
        for title, content in sections.items():
            if "conclusion" in title.lower():
                return summarize_text(content, 3)

    if "keyword" in question:
        return "Top Keywords: " + ", ".join(full_res["keywords"])

    if "domain" in question:
        return f"This paper belongs to the domain: {full_res['domain']}"

    if "publisher" in question:
        return f"Publisher detected: {full_res['publisher']}"

    if "score" in question:
        return f"Composite Score: {full_res['scores']['Composite']}"

    if "word count" in question:
        return f"Total Word Count: {full_res['stats']['word_count']}"

    best_match = ""
    max_score = 0

    for title, content in sections.items():
        score = sum(1 for word in question.split() if word in content.lower())

        if score > max_score:
            max_score = score
            best_match = content

    if max_score < 2:   
        return "Sorry, I can’t find relevant information in this paper for that question."

    return summarize_text(best_match, 3)

STOPWORDS = set([
    "the","is","in","and","to","of","for","on","with",
    "a","an","by","this","that","it","as","are","was"
])

def detect_repetition(text, threshold=5):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    words = [w for w in words if w not in STOPWORDS]

    word_counts = Counter(words)

    repeated = {word: count for word, count in word_counts.items() if count >= threshold}

    return repeated
def generate_structured_suggestions(text):

    suggestions = []
    sentences = re.split(r'[.!?]', text)
    long_sentences = [s for s in sentences if len(s.split()) > 30]

    if long_sentences:
        suggestions.append("• Consider breaking long sentences into shorter ones for clarity.")

    repeated_words = detect_repetition(text)

    if repeated_words:
        suggestions.append("• Reduce repetition of words like: " +
                           ", ".join(list(repeated_words.keys())[:5]))

    if " was " in text or " were " in text:
        suggestions.append("• Review passive voice usage for stronger academic tone.")

    if not suggestions:
        suggestions.append("• Writing structure looks good. Minor refinements may improve clarity.")

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
        "Oxford University Press": ["oxford university press"]
    }

    text_lower = text.lower()

    for publisher, keywords in publishers.items():
        for word in keywords:
            if word in text_lower:
                return publisher

    return "Not Detected"


st.markdown(""" 
<style>

.stApp {
    background: linear-gradient(135deg, #f4f6fb, #eef1f7);
    font-family: 'Segoe UI', sans-serif;
}

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
    box-shadow: 0px 6px 15px rgba(0,0,0,0.2);
}

[data-testid="stFileUploader"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

button[data-baseweb="tab"] {
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #6a11cb !important;
    border-bottom: 3px solid #6a11cb !important;
}

.summary-box {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 5px solid #6a11cb;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
    transition: 0.3s ease;
}

.summary-box:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 22px rgba(0,0,0,0.1);
}

.keyword-tag {
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    padding: 6px 12px;
    margin: 5px;
    border-radius: 20px;
    display: inline-block;
    font-size: 13px;
}

div[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #6a11cb !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

div[data-testid="stTextInput"] input:focus {
    border: 2px solid #2575fc !important;
    box-shadow: 0px 0px 8px rgba(106,17,203,0.4);
}

.stAlert {
    border-radius: 10px;
}
            /* Remove top white space */
header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
}

.block-container {
    padding-top: 1rem !important;
}

/* Remove extra white bar */
[data-testid="stToolbar"] {
    display: none;
}


hr {
    border: none;
    height: 1px;
    background: #e0e0e0;
    margin: 20px 0;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("User Dashboard")
    st.write(f"👤 {st.session_state.user_data['username']}")
    st.write(f"🎓 Role: {st.session_state.user_data['role']}")
    st.markdown("---")

    if st.button("Logout", key="sidebar_logout"):
        st.session_state.signed_up = False
        st.rerun()


st.markdown("### 📤 Upload Research Paper")
uploaded_file = st.file_uploader("", type=["pdf"])

if uploaded_file:
    if st.button("Analyze Document", type="primary", key="analyze_btn"):
        with st.spinner("Analyzing document..."):
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            

            cleaned_text = clean_text(text)
            results = analyze_full_document(cleaned_text)

            if results is not None:

                keywords, domain = extract_keywords_and_domain(cleaned_text)
                publisher = detect_publisher(cleaned_text)
                title = detect_title(cleaned_text)

                results["keywords"] = keywords
                results["domain"] = domain
                results["publisher"] = publisher
                results["title"] = title
                results["full_text"] = cleaned_text

                st.session_state['results'] = results
                st.session_state['filename'] = uploaded_file.name

            else:
                st.error("Unable to analyze document.")

if 'results' in st.session_state:
    

    res = st.session_state['results']
    scores = res['scores']
    stats = res['stats']

    st.markdown("## 📄 Research Paper Title")
    st.info(res.get("title", "Not detected"))

    st.markdown("## 📈 Analysis Results")

    grade = get_grade(scores['Composite'])

    col_main1, col_main2 = st.columns([2,1])

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
        pdf_report = generate_pdf_report(res, st.session_state['filename'])
        st.download_button(
            "Download Full Report",
            data=pdf_report,
            file_name="PaperIQ_Report.pdf",
            mime="application/pdf"
        )



    tab1, tab2, tab3, tab4, tab5 ,tab6 ,tab7 = st.tabs([
        "📊 Visualizations",
        "📑 Section Summaries",
        "📄 Metadata",
        "💡 Suggestions",
        "❤️ Sentiment",
        "📌 Keywords & Domain",
        "Chatbot"
    ])

    with tab1:
        st.subheader("📊 Performance Breakdown")

        categories = ['Language', 'Coherence', 'Reasoning', 'Sophistication', 'Readability']
        values = [scores[c] for c in categories]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            text=values,
            textposition='auto'
        ))

        fig.update_layout(
            title="Score Distribution (Out of 100)",
            yaxis=dict(title="Score", range=[0, 100]),
            xaxis=dict(title="Metrics"),
            height=400
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
            "Select Summary Length",
            ["Short", "Medium", "Long"],
            horizontal=True
        )

        if summary_length == "Short":
            num_sentences = 2
        elif summary_length == "Medium":
            num_sentences = 4
        else:
            num_sentences = 6

        for title, content in res['sections'].items():

            summary = summarize_text(content, num_sentences)
            important_sentences = get_important_sentences(content, num_sentences)

            highlighted_text = content

            for sent in important_sentences:
                highlighted_text = highlighted_text.replace(
                    sent,
                    f"<mark style='background-color:#fff59d; padding:2px;'>{sent}</mark>"
                )

            st.markdown(f"""
            <div class="summary-box">
                <h4>📌 {title}</h4>
                <p><b>Summary:</b> {summary}</p>
                <hr>
                <p><b>Highlighted Important Sentences:</b></p>
                <p>{highlighted_text}</p>
            </div>
            """, unsafe_allow_html=True)


    with tab3:
        st.write(f"**Publisher:** {res['publisher']}")
        st.write(f"**File Name:** {st.session_state['filename']}")
        st.write(f"**Words:** {stats['word_count']}")
        st.write(f"**Sentences:** {stats['sentence_count']}")

    with tab4:
        st.subheader("💡 Writing Suggestions")

        suggestions = generate_structured_suggestions(res['full_text'])

        suggestion_html = "<br>".join(suggestions)

        st.markdown(f"""
        <div class="summary-box">
            {suggestion_html}
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.subheader("📊 Section-wise Sentiment Analysis")

        sentiment_data = section_wise_sentiment(res['sections'])

        for section, data in sentiment_data.items():
            st.markdown(f"""
            <div class="summary-box">
                <h4>{section}</h4>
                <p><b>Sentiment:</b> {data['label']}</p>
            </div>
            """, unsafe_allow_html=True)
    with tab6:
        st.subheader("📌 Extracted Keywords")

        keyword_html = " ".join([
            f"<span style='background-color:#d9d9d9; padding:6px 10px; margin:5px; border-radius:15px; display:inline-block;'>{kw}</span>"
            for kw in res['keywords']
        ])

        st.markdown(keyword_html, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📚 Detected Domain")
        st.success(res['domain'])
    with tab7:
        st.subheader("🤖 Ask About This Paper")

        user_question = st.text_input(
    "Ask a question about the paper",
    key="paper_chatbot_input"
)

        if user_question:
            answer = paper_chatbot(user_question, res['sections'], res)
            st.markdown(f"""
            <div class="summary-box">
            <b>Answer:</b> {answer}
            </div>
            """, unsafe_allow_html=True) 

