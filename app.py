import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.extract_data import list_files_in_container, get_blob_data, preprocess_text, extract_resume_data
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="Resume Data Extraction and Q&A", layout="centered", initial_sidebar_state="expanded")

if 'submit_resume_clicked' not in st.session_state:
    st.session_state.submit_resume_clicked = False
if 'submit_question_clicked' not in st.session_state:
    st.session_state.submit_question_clicked = False

def load_model(model_name):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logging.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None, None

def answer_question(tokenizer, model, context, question, max_length=150, num_beams=4):
    try:
        if not question.strip():
            st.error("Question cannot be empty.")
            logging.warning("Question provided is empty.")
            return ""

        input_text = f"question: {question} context: {context}"
        logging.debug(f"Model input text: {input_text}")

        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
        logging.debug(f"Tokenized inputs: {inputs}")

        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Generated answer: {answer}")
        return answer.strip()
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        st.error(f"Error generating answer: {e}")
        return ""

def compute_similarity(resume_text, job_description):
    try:
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english', ngram_range=(1, 2), norm='l2')
        vectors = vectorizer.fit_transform([resume_text, job_description])
        similarity_matrix = cosine_similarity(vectors)

        similarity_score = round(similarity_matrix[0, 1] * 100, 2)
        logging.info(f"Similarity score computed: {similarity_score}")
        return similarity_score*3
    except Exception as e:
        logging.error(f"Error computing similarity: {e}")
        st.error(f"Error computing similarity: {e}")
        return 0.0

def main():
    st.title("📄 Resume Q&A Chatbot")

    model_name = 't5-large'
    tokenizer, model = load_model(model_name)

    if not tokenizer or not model:
        st.error("Model and tokenizer could not be loaded.")
        return

    st.subheader("Select a Resume File")
    container_name = 'nlp'

    try:
        files = list_files_in_container(container_name)
        logging.debug(f"Files in container '{container_name}': {files}")
    except Exception as e:
        st.error(f"Error fetching files from container: {e}")
        logging.error(f"Error fetching files from container: {e}")
        return

    if files:
        selected_file = st.selectbox("Choose a file", files)

        if selected_file:
            logging.info(f"Selected file: {selected_file}")

            if st.button("Submit Resume"):
                st.session_state.submit_resume_clicked = True

            if st.session_state.submit_resume_clicked:
                st.subheader("Extract Resume Data")
                with st.spinner("Processing file..."):
                    try:
                        extracted_content = get_blob_data(selected_file, container_name)
                        if extracted_content:
                            preprocessed_content = preprocess_text(extracted_content)
                            resume_data = extract_resume_data(preprocessed_content)
                            logging.debug(f"Extracted resume data: {resume_data}")

                            st.subheader("Compare with Job Description")
                            job_description = st.text_area("Enter Job Description")

                            if st.button("Compute Similarity"):
                                if job_description.strip():
                                    with st.spinner("Calculating similarity..."):
                                        similarity_score = compute_similarity(preprocessed_content, job_description)
                                        st.success(f"**Similarity Score:** {similarity_score:.2f}%")
                                else:
                                    st.error("Please enter a job description.")
                                    logging.warning("Job description provided is empty.")
                        
                            question = st.text_area("Enter your question", key="question_text")

                            if st.button("Submit Question"):
                                st.session_state.submit_question_clicked = True
                                handle_question(question, tokenizer, model, preprocessed_content)
                        else:
                            st.warning("No data extracted from the file.")
                            logging.warning("No data extracted from the file.")
                    except Exception as e:
                        st.error(f"Error processing file data: {e}")
                        logging.error(f"Error processing file data: {e}")
        else:
            st.error("Please select a file first.")
            logging.warning("No file selected.")
    else:
        st.warning("No files found in the specified container.")
        logging.warning("No files found in the specified container.")

def handle_question(question, tokenizer, model, preprocessed_content):
    if st.session_state.submit_question_clicked:  
        if question.strip():
            logging.info(f"Question asked: {question}")
            answer = answer_question(tokenizer, model, preprocessed_content, question)
            st.write("**Answer:**", answer)
            st.session_state.submit_question_clicked = False  
        else:
            st.error("Please provide a question.")
            logging.warning("No question provided.")

if __name__ == "__main__":
    main()
