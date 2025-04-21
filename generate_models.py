import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download if not already done
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-cleaned datasets (Pickle files)
df = pd.read_pickle('candidate_profiles.pkl')
df1 = pd.read_pickle('job_postings.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Train and save recruiter vectorizer
tfidf_recruiter = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix_r = tfidf_recruiter.fit_transform(df['combined_features'])

with open('tfidf_recruiter.pkl', 'wb') as f:
    pickle.dump(tfidf_recruiter, f)

# Train and save candidate vectorizer
tfidf_candidate = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix_c = tfidf_candidate.fit_transform(df1['combined_features'])

with open('tfidf_candidate.pkl', 'wb') as f:
    pickle.dump(tfidf_candidate, f)

print("✅ TF-IDF models saved successfully!")
