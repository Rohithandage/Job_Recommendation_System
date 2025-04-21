import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load your CSVs
df = pd.read_csv(r"C:\Users\rohit\OneDrive\Desktop\Job_sample_dataset_13.csv").dropna()
df1 = pd.read_csv(r"C:\Users\rohit\OneDrive\Desktop\Jobs_sample_dataset _12.csv").dropna()

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

# Recruiter (df)
df['combined_features'] = (
    df['Skills'].apply(clean_text) + " " +
    df['Name'].apply(clean_text) + " " +
    df['Job_title'].apply(clean_text) + " " +
    df['Experience'].apply(clean_text) + " " +
    df['Work_Location'].apply(clean_text) + " " +
    df['Work_Preference'].apply(clean_text) + " " +
    df['Company'].apply(clean_text)
)

# Candidate (df1)
df1['combined_features'] = (
    df1['Skills'].apply(clean_text) + " " +
    df1['Job_title'].apply(clean_text) + " " +
    df1['Required_experience'].apply(clean_text) + " " +
    df1['Location'].apply(clean_text) + " " +
    df1['Preferences'].apply(clean_text) + " " +
    df1['Company'].apply(clean_text)
)

# Save the DataFrames as Pickles
df.to_pickle('candidate_profiles.pkl')
df1.to_pickle('job_postings.pkl')

print("✅ Pickle files created successfully!")
