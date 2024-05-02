from flask import Flask, render_template, request
import pandas as pd
import PyPDF2
from pyresparser import ResumeParser
from sklearn.neighbors import NearestNeighbors
from src.components.job_recommender import ngrams,getNearestN,jd_df
import src.skills_extraction as skills_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def process_resume(file_path):
    resume_skills=skills_extraction.skills_extractor(file_path)

    skills=[]
    skills.append(' '.join(word for word in resume_skills))
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)

    
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
    jd_test = (jd_df['Processed_JD'].values.astype('U'))

    distances, indices = getNearestN(jd_test)
    test = list(jd_test) 
    matches = []

    for i,j in enumerate(indices):
        dist=round(distances[i][0],2)
        temp = [dist]
        matches.append(temp)
    
    matches = pd.DataFrame(matches, columns=['Match confidence'])
    jd_df['match']=matches['Match confidence']
    
    return jd_df.head(5).sort_values('match')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        uploaded_file = request.files['resume']
        
        if uploaded_file.filename != '':
            file_path = 'uploads/' + uploaded_file.filename
            uploaded_file.save(file_path)
            df_jobs = process_resume(file_path)
            return render_template('result.html', jobs=df_jobs.to_html())
        else:
            return "No file selected"

if __name__ == '__main__':
    app.run(debug=True)
