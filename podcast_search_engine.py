import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import json

# Load dataset from JSON file
with open("data/podcasts.json", "r", encoding="utf-8") as f:
    podcasts_online = json.load(f)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u0600-\u06FF\w\s-]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens_stemmed = [stemmer.stem(word) for word in tokens]
    # tokens_lemmatized = [lemmatizer.lemmatize(word) for word in tokens_stemmed]
    return ' '.join(tokens_stemmed)


def search(query, descriptions, indices):
    

    query = preprocess_text(query)
    if not query.strip():
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(similarity_scores)[::-1]
    ranked_scores = similarity_scores[ranked_indices]

    results = [(indices[i], ranked_scores[idx]) 
               for idx, i in enumerate(ranked_indices) 
               if ranked_scores[idx] > 0]
    return results

podcast_titles = []
podcast_descriptions = []
podcast_urls = []

for podcast in podcasts_online:
    full_text = f"{podcast['title']} {podcast['description']}"
    cleaned = preprocess_text(full_text)
    if cleaned.strip():
        podcast_titles.append(podcast['title'])
        podcast_descriptions.append(cleaned)
        podcast_urls.append(podcast['url'])

# =============================================================================
import streamlit as st

st.set_page_config(page_title="Podcast Search engine",page_icon="🔎")
st.title(" Podcast Search Engine")
st.markdown(
   """
    <style>
    .stApp {

        background-image: url("https://img.freepik.com/premium-photo/studio-podcast-microphone-dark-background_162008-316.jpg?w=1380");
        background-size: 110% 110%;
        background-position: 110% 50% ;
        opacity: 0.8;
        background-repeat: no-repeat;
    }

    
    header {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
)
query = st.text_input("", placeholder=" أبهرني عايز تشوف اي ")
query = query.strip()
if query:
    
    results = search(query, podcast_descriptions, list(range(len(podcast_titles))))

    if results:
        st.success(f"تم العثور على {len(results)} نتيجة:")
        
        relevant_flags = []
        for idx, score in results:
            st.subheader(f"🎧 Title:  {podcast_titles[idx]}")
            st.markdown(f"🔗 **Link** :  {podcast_urls[idx]}")
            st.markdown(f"**📝 Description:**  {podcasts_online[idx]['description']}")
            st.markdown(f"✅ **Similarity Score**:  {round(score * 100, 2)}٪")
            
            marked = st.checkbox("Mark as relevant", key=podcast_titles[idx])
            relevant_flags.append(marked)
            st.divider()
            # st.markdown("---")
        relevant_count = sum(relevant_flags)
        ratio = relevant_count / len(results)
        st.info(f"📊 Precision: {round(ratio, 2)}")
    else:
        st.warning("لم يتم العثور على نتائج مطابقة.")
else:
    st.info("يرجى إدخال عبارة للبحث.")