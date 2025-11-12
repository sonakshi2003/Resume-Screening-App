                  
# .\venv\Scripts\activate                 

  
# streamlit run app.py                   

import streamlit as st
import pickle
import docx
import PyPDF2
import re
import pandas as pd

# Load model & vectorizer & encoder
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    # Dark theme CSS + styling
    st.markdown(
        """
        <style>
        /* Dark background and text */
        .reportview-container {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Header styling */
        .title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 20px;
            color: #03dac6;
        }
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1f1f1f;
            color: #e0e0e0;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #03dac6;
            color: #121212;
            font-weight: bold;
            font-size: 16px;
            height: 3em;
            width: 100%;
            border-radius: 10px;
            transition: background-color 0.3s ease;
            margin-top: 1em;
        }
        div.stButton > button:hover {
            background-color: #018786;
            cursor: pointer;
        }
        /* Download button */
        div.stDownloadButton > button {
            background-color: #bb86fc;
            color: #121212;
            font-weight: bold;
            font-size: 16px;
            height: 3em;
            width: 100%;
            border-radius: 10px;
            margin-top: 1em;
            transition: background-color 0.3s ease;
        }
        div.stDownloadButton > button:hover {
            background-color: #7f39fb;
            cursor: pointer;
        }
        /* Text area in dark mode */
        textarea {
            background-color: #2c2c2c;
            color: #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["About", "Prediction"])

    # Logo image above title (replace with your own URL or local file)
    st.image("logo.png", width=120)

    st.markdown('<h1 class="title">Resume Category Prediction App</h1>', unsafe_allow_html=True)

    if app_mode == "About":
        st.header("About This App")
        st.write(
            """
           

### ✅ About This Application

Welcome to the **Resume Category Prediction App**!

This AI-powered application helps you automatically **classify resumes** into job categories like **Data Science, Web Development, HR, Testing, DevOps**, and more — based on the resume's content.

---

### 🔍 Key Features:

* **Multi-format support**: Upload resumes in **PDF**, **DOCX**, or **TXT** formats.
* **Batch predictions**: Upload and analyze **multiple resumes** at once.
* **Text preview**: View extracted text from each resume before or after prediction.
* **AI-powered classification**: Uses machine learning (SVM + TF-IDF) to predict resume categories.
* **CSV Export**: Download all predictions in one click as a CSV file.
* **Attractive dark theme**: Modern, sleek, and user-friendly interface.

---

### 🤖 Behind the Scenes:

* **Text Extraction**: PyPDF2 for PDFs, python-docx for DOCX, and UTF-8 decoding for TXT.
* **Text Cleaning**: Removes URLs, symbols, special characters, etc.
* **TF-IDF Vectorization**: Converts cleaned text into numeric format.
* **SVM Classifier**: Predicts the job category based on resume content.
* **Label Encoder**: Decodes category labels to human-readable names.

---

### 🎯 Use Case:

This tool is ideal for:

* **Recruiters & HR teams** who want to save time categorizing incoming resumes.
* **Hiring platforms** looking to automate resume parsing and tagging.
* **Job seekers** who want to understand how their resume may be perceived by ML-based filters.

---

### 🧠 Tip:

Upload your resumes in the **Prediction** tab and click **"Predict Categories"**. You can preview the resume content and download all the results as a CSV.

---

            """
        )

    elif app_mode == "Prediction":
        st.header("Upload Resumes & Predict Categories")
        uploaded_files = st.file_uploader(
            "Upload one or more resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="You can upload multiple files at once.",
        )

        all_results = []

        if uploaded_files:
            if st.button("Predict Categories"):
                for uploaded_file in uploaded_files:
                    try:
                        text = handle_file_upload(uploaded_file)
                        category = pred(text)
                        all_results.append(
                            {"Resume Name": uploaded_file.name, "Predicted Category": category}
                        )
                    except Exception as e:
                        all_results.append(
                            {"Resume Name": uploaded_file.name, "Predicted Category": f"Error: {str(e)}"}
                        )

                df_results = pd.DataFrame(all_results)
                st.success("Prediction complete! Here are the results:")
                st.dataframe(df_results, use_container_width=True)

                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download All Predictions as CSV",
                    data=csv,
                    file_name="resume_predictions.csv",
                    mime="text/csv",
                )

            if st.checkbox("Show extracted text previews for all uploaded resumes"):
                for uploaded_file in uploaded_files:
                    try:
                        with st.expander(f"Preview: {uploaded_file.name}", expanded=False):
                            preview_text = handle_file_upload(uploaded_file)
                            st.text_area(f"Extracted text from {uploaded_file.name}", preview_text, height=300)
                    except Exception as e:
                        st.error(f"Cannot preview {uploaded_file.name}: {e}")

if __name__ == "__main__":
    main()

