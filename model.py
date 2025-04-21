import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from sklearn.metrics.pairwise import linear_kernel

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_pickle('candidate_profiles.pkl')
df1 = pd.read_pickle('job_postings.pkl')

with open('tfidf_recruiter.pkl', 'rb') as f:
    tfidf_recruiter = pickle.load(f)
with open('tfidf_candidate.pkl', 'rb') as f:
    tfidf_candidate = pickle.load(f)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def get_recommendation(Skills, Experience=None, top_n=10):
    cleaned_skills = clean_text(Skills).split()
    skill_mask = df['Skills'].apply(lambda x: all(skill in clean_text(x) for skill in cleaned_skills))
    filtered_jobs = df[skill_mask]

    if Experience:
        cleaned_exp = clean_text(Experience)
        exp_mask = filtered_jobs['Experience'].apply(lambda x: cleaned_exp in clean_text(x))
        filtered_jobs = filtered_jobs[exp_mask]

    if len(filtered_jobs) == 0:
        return 'No matching jobs found with these criteria'

    user_query = clean_text(Skills)
    if Experience:
        user_query += ' ' + clean_text(Experience)

    query_vec = tfidf_recruiter.transform([user_query])
    filtered_matrix = tfidf_recruiter.transform(filtered_jobs['combined_features'])
    cosine_sim = linear_kernel(query_vec, filtered_matrix).flatten()
    sim_indices = cosine_sim.argsort()[-top_n:][::-1]

    return filtered_jobs[['Job_title', 'Name', 'Experience', 'Work_Location', 'Company', 'Skills']].iloc[sim_indices]

def get_job_recommendations(Skills, top_n=10):
    cleaned_skills = clean_text(Skills).split()
    skill_mask = df1['Skills'].apply(lambda x: all(skill in clean_text(x) for skill in cleaned_skills))
    filtered_jobs = df1[skill_mask]

    if len(filtered_jobs) == 0:
        return 'No matching jobs found with these skills and location'

    user_query = clean_text(Skills)
    query_vec = tfidf_recruiter.transform([user_query])
    filtered_matrix = tfidf_recruiter.transform(filtered_jobs['combined_features'])
    cosine_sim = linear_kernel(query_vec, filtered_matrix).flatten()
    sim_indices = cosine_sim.argsort()[-top_n:][::-1]

    return filtered_jobs[['Job_title', 'Location', 'Company', 'Required_experience', 'Skills']].iloc[sim_indices]
