# 🎧 Podcast Search Engine
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLTK-154E98?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white"/>
</p>

<p align="center">
  <!-- Repo Status Badges -->
  <img src="https://img.shields.io/github/last-commit/El-Qady/Podcast-Search-Engine"/>
  <img src="https://img.shields.io/github/languages/count/El-Qady/Podcast-Search-Engine"/>
  <img src="https://img.shields.io/github/repo-size/El-Qady/Podcast-Search-Engine"/>
  <img src="https://img.shields.io/github/license/El-Qady/Podcast-Search-Engine"/>
  <img src="https://img.shields.io/github/stars/El-Qady/Podcast-Search-Engine?style=social"/>
</p>
## 📌 Description
A search engine application designed to find relevant podcasts based on user queries.  
The system leverages **Natural Language Processing (NLP)** techniques to preprocess podcast metadata and retrieve the most relevant matches using **TF-IDF** and **Cosine Similarity**.

---

## 🚀 Key Steps & Features

### 🔹 Data Preparation
- Dataset of podcasts stored in **`data/podcasts.json`** (title, description, url).
- Titles and descriptions combined for indexing.

### 🔹 Text Preprocessing
- Lowercasing, digit & punctuation removal, whitespace normalization.  
- Tokenization, stopword removal, and stemming using **NLTK**.

### 🔹 Search & Ranking
- Implemented **TF-IDF Vectorization** to represent podcast descriptions.  
- Used **Cosine Similarity** to measure query relevance.  
- Results ranked in descending order of similarity score.

### 🔹 User Interface
- Developed an **interactive web app** using **Streamlit**.  
- Real-time search with instant results (titles, descriptions, links, similarity scores).  
- Added **"Mark as Relevant"** feature to calculate **Precision metric** dynamically.

### 🔹 UI Design Enhancements
- Custom background styling with **CSS inside Streamlit**.  
- Clean, minimal, and responsive layout.

---

## 🛠 Technologies Used
- **Programming Language**: Python  
- **Libraries & Tools**: NLTK, NumPy, scikit-learn (TfidfVectorizer, cosine_similarity), Streamlit  
- **Data Format**: JSON (`data/podcasts.json`)  
- **Text Processing**: Tokenization, stopword removal, stemming  
- **Search Algorithm**: TF-IDF + Cosine Similarity  
- **UI Styling**: Custom CSS inside Streamlit app  

---

## ⚡ Highlights
- Real-time podcast search with **semantic text matching**.  
- Integrated **Precision metric** directly into the UI.  
- Fully interactive interface with instant feedback.  
- Easy-to-update dataset in **JSON format**.  

---

## 📂 Project Structure
```
podcast_search_engine/
│── app.py               # Main Streamlit application
│── utils.py             # Helper functions (text preprocessing, search, etc.)
│── requirements.txt     # Project dependencies
│── README.md            # Project documentation
│── data/
│   └── podcasts.json    # Dataset of podcasts
```

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Upload your queries and get relevant podcast results instantly! 🎧  

---

## ✨ Future Improvements
- Add support for larger datasets (API integration).  
- Include advanced ranking methods (e.g., BERT embeddings).  
- Implement user history & favorites feature.  
