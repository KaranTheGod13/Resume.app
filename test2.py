import os
import pdfplumber
import spacy
import pytesseract
import docx
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def rank_resumes(resumes, job_description):
    processed_resumes = [preprocess_text(res) for res in resumes]
    processed_job_desc = preprocess_text(job_description)
    
    vectorizer = TfidfVectorizer()
    all_texts = processed_resumes + [processed_job_desc]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    job_desc_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(resume_vectors, job_desc_vector.reshape(1, -1))
    
    ranked_indices = np.argsort(similarity_scores.flatten())[::-1]
    return ranked_indices, similarity_scores.flatten()

# Streamlit UI
st.title(" 📄 AI-Powered Resume Screening")
st.write("Upload multiple resumes (PDF, DOCX, JPG) and get ranked results!")

job_description = st.text_area(" ✍️ Enter Job Description:")
uploaded_files = st.file_uploader(" 📂 Upload Resumes", type=["pdf", "docx", "jpg", "png"], accept_multiple_files=True)

if st.button("Analyze Resumes") and uploaded_files and job_description:
    resume_texts = []
    file_names = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_names.append(file_name)
        
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif file_name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)
        elif file_name.endswith(".jpg") or file_name.endswith(".png"):
            text = extract_text_from_image(uploaded_file)
        else:
            text = ""
        
        resume_texts.append(text)
    
    ranked_indices, scores = rank_resumes(resume_texts, job_description)
    
    st.subheader("Top Ranked Resumes")
    for i, index in enumerate(ranked_indices[:10]):
        st.write(f"{i+1}. {file_names[index]} - Score: {scores[index]:.4f}")
