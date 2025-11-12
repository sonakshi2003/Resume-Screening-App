# Resume-Screening-App
The Resume Screening App is an AI-based web application built using Streamlit that automatically classifies resumes into relevant job categories such as Data Science, Web Development, Testing, and more. The system streamlines recruitment by reducing manual screening time through intelligent text extraction and machine learning based classification.
Key Features:
1.Supports multiple file formats: PDF, DOCX, TXT
2.Extracts and cleans resume text using PyPDF2 and python-docx
3.Uses TF-IDF vectorization and a Support Vector Machine (SVM) classifier for text-based job category prediction
4.Displays category predictions with an option to download results as CSV
5.Includes preview functionality for extracted resume text
Technical Stack:
1.Frontend/UI: Streamlit
2.Backend/ML: Python, scikit-learn, SVM, TF-IDF
3.Libraries: PyPDF2, python-docx, pandas, re, pickle
4.Model Assets: clf.pkl (SVM model), tfidf.pkl (vectorizer), encoder.pkl (label encoder)
Impact:
This tool automates the first stage of resume screening, helping recruiters efficiently categorize and shortlist candidates while saving hours of manual effort. It can be easily integrated into larger ATS (Applicant Tracking Systems) or HR platforms.
