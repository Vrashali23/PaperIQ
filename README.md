# 📄 PaperIQ – AI Powered Research Insight Analyzer

PaperIQ is an intelligent **AI-powered research paper analysis platform** built using **Python and Streamlit**.
It helps researchers, students, and faculty quickly analyze research papers and extract meaningful insights such as summaries, keywords, research gaps, sentiment analysis, and citation statistics.

The system simplifies academic reading by automatically evaluating paper quality and generating detailed analysis reports.

---

# 🚀 Features

## 📊 Research Paper Evaluation

* Composite research quality score
* Language quality analysis
* Coherence and reasoning evaluation
* Readability score
* Vocabulary sophistication analysis

---

## 📑 Section-wise Smart Summarization

PaperIQ automatically detects sections such as:

* Abstract
* Introduction
* Literature Review
* Methodology
* Results
* Discussion
* Conclusion

Each section is summarized using **NLP-based summarization algorithms**.

---

## 📌 Keyword Extraction & Domain Detection

The system extracts the most important keywords and identifies the research domain such as:

* Computer Science
* Engineering
* Healthcare
* Finance
* Education

---

## 🧠 Advanced Research Insights

PaperIQ performs deeper analysis including:

* Research gap detection
* Contribution detection
* Semantic strength scoring
* Novelty score calculation
* Internal redundancy detection

---

## 📚 Citation Analysis

The system automatically identifies citation patterns and calculates:

* Total citations
* Citation density
* Impact score
* Average citation year

---

## 🤖 Research Paper Chatbot

Users can ask questions about the uploaded research paper.

The chatbot uses **TF-IDF vectorization and cosine similarity** to retrieve the most relevant answers from the paper content.

---

## 📊 Paper Comparison

PaperIQ supports comparison of multiple research papers including:

* Composite score comparison
* Citation comparison
* Semantic strength comparison
* Paper similarity matrix

---

## 📈 Visual Analytics

Interactive visualizations include:

* Score distribution graphs
* Performance breakdown charts
* Paper comparison charts

---

## 📂 Analysis History

Users can view previously analyzed papers and download reports again.

---

## 📄 Automatic PDF Report Generation

PaperIQ generates a detailed PDF report including:

* Document statistics
* Score breakdown
* Section summaries
* Visual charts
* Research insights

---

# 🛠 Tech Stack

### Programming Language

* Python

### Framework

* Streamlit

### Libraries Used

* NumPy
* TextBlob
* PDFPlumber
* python-docx
* Plotly
* Scikit-learn
* FPDF
* Pandas
* Matplotlib
* Regular Expressions (re)

These libraries provide functionality for **Natural Language Processing, Machine Learning, Data Visualization, and Document Processing**.

---

# 📂 Project Structure

```
PaperIQ
│
├── app.py
├── README.md
├── requirements.txt
│
├── charts
│   └── Generated analysis charts
│
├── reports
│   └── Generated PDF reports
```

---

# ▶️ Installation

### 1️⃣ Clone the Repository

```
git clone https://github.com/Vrashali23/PaperIQ.git
```

---

### 2️⃣ Navigate to Project Folder

```
cd PaperIQ
```

---

### 3️⃣ Install Required Dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# 📊 How PaperIQ Works

1. User uploads a research paper (PDF format).

2. The system extracts text using PDF processing libraries.

3. NLP algorithms analyze the document content.

4. PaperIQ generates insights including:

   * Section summaries
   * Keywords and domain
   * Citation analysis
   * Research gaps
   * Sentiment analysis
   * Paper quality scores

5. A complete **PDF analysis report** is generated.

---

# 📈 Example Insights Generated

PaperIQ provides outputs such as:

* Composite Research Quality Score
* Language and Readability Evaluation
* Research Domain Detection
* AI Writing Feedback
* Recommended Academic Journals
* Paper Similarity Comparison
* Research Gap Identification

---

# 📌 Future Improvements

* AI research assistant integration
* Plagiarism detection
* Improved section detection
* Cloud deployment
* User authentication with database
* Citation network visualization

---

# Contributing

```bash
# get latest changes 
git pull origin main 

# make your changes 
git add . 

git commit -m "short description" 

# push to branch/main 
git push origin main
```

> [!NOTE] 
> Open a Pull Request
> Create a PR from your branch to the `main` branch of this repository.

---

# 📜 License

This project is intended for **educational and research purposes**.
