import spacy
from spacy.matcher import Matcher
import PyPDF2
import os
import pandas as pd 

nlp = spacy.load('en_core_web_sm')

file_path=r'job_decription.csv'
with open(file_path, 'r') as file:
    csv_reader = pd.read_csv(file)
    skills = [skill for skill in csv_reader['skills']]

skill_patterns = [[{'LOWER': skill}] for skill in skills[0]]

matcher = Matcher(nlp.vocab)

for pattern in skill_patterns:
    matcher.add('Skills', [pattern])
    
def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills

def extract_text_from_pdf(file_path:str):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def skills_extractor(file_path):
        path=r''
        full_file_path = os.path.join(path, file_path)
        resume_text = extract_text_from_pdf(full_file_path)
        skills = list(extract_skills(resume_text))
        return skills