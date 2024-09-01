import streamlit as st
import pdfplumber
import csv
import questions
import execute
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from tqdm import tqdm

def main():
    st.title("QuesGen")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False, key="upload_pdf")

    # Language model selection menu
    model_options = ["GPT-3.5-turbo", "GPT-4o"]
    selected_model = st.selectbox("Select Language Model", model_options)

    st.markdown(
            """
            <div style="border: 2px solid #ffaa33; padding: 10px; border-radius: 10px;">
                <h3 style="color: #ffaa33;">One-click execution</h3>
                <p>Generate questions and output them to a file.</p>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("One-click execution"):
        if uploaded_file is None:
            st.error("Please upload a PDF file")
        else:
            progress_bar = st.progress(0)
            questions.main(uploaded_file.name, progress_bar)
            st.success("Questions have been generated and saved to Q_output.txt")
            progress_text = "Generating answers..., please wait for a while..."
            progress_bar = st.progress(0, text=progress_text)
            answer(uploaded_file.name, "Q_output.txt", progress_bar)


    st.markdown(
            """<div style="border: 2px solid #00bbff; padding: 10px; border-radius: 10px;">
                <h3 style="color: #00bbff;">Execute step by step</h3>
                <p>You can execute the programs for generating questions and answering questions separately.</p>"""
                , unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
                <h3><strong>Question Generation</strong></h3>
                <p>Generate questions and output them to a file.</p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        uploaded_pdf_Q = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False, key="upload_pdf_Q")
        if st.button("Generate Question"):
            if uploaded_pdf_Q is None:
                st.error("Please upload a PDF file")
            else:
                progress_bar = st.progress(0)
                questions.main(uploaded_pdf_Q.name, progress_bar)
                st.success("Questions have been generated and saved to Q_output.txt")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(
            """
                <h3>Answer Generation</h3>
                <p>Generate answers and output them to a file.</p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        uploaded_pdf_A = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False, key="upload_pdf_A")
        uploaded_txt = st.file_uploader("Upload TXT file", type=["txt"], accept_multiple_files=False)
        if st.button("Generate Answer"):
            if uploaded_pdf_A is None:
                st.error("Please upload a PDF file")
            elif uploaded_txt is None:
                st.error("Please upload a TXT file")
            else:
                progress_text = "Generating answers..., please wait for a while..."
                progress_bar = st.progress(0, text=progress_text)
                answer(uploaded_pdf_A.name, uploaded_txt.name, progress_bar)
        st.markdown("</div>", unsafe_allow_html=True)

def answer(uploaded_file, question_path, progress_bar):
    question = []
    second_answer = []
    if uploaded_file is not None:
        docs = execute.get_reference_document(uploaded_file, 800, 200)
        question = execute.get_question_from_txt(question_path)

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        vector_store = Chroma.from_documents(docs, embeddings)
        print(len(question))

        
        for i in tqdm(range(0, len(question)), desc="Processing questions"):
            messages = execute.replace_message_content(question[i])
            generated_tokens_1, generated_probabilities_1 = execute.first_retrieval_answer(messages, vector_store, question, i)
            generated_tokens_1 = execute.remove_below_threshold_tokens(generated_tokens_1, generated_probabilities_1, 0.5, 1)

            messages = execute.replace_message_content(question[i])
            generated_tokens_2, generated_probabilities_2, second_answer = execute.second_retrieval_answer(second_answer, messages, vector_store, generated_tokens_1, question, i-1)
            progress_text = "Generating answers..., please wait for a while..."
            progress_bar.progress((i+1) / len(question), text=progress_text)
            # execute.remove_below_threshold_tokens(generated_tokens_2, generated_probabilities_2, 0.5)

        # Save the output file as CSV

        output_filename = "Answer.csv"
        with open(output_filename, "w", newline='', encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Question", "Answer"])
            for i in range(len(second_answer)):
                csvwriter.writerow([question[i], second_answer[i]])

        st.success(f"The answers have been generated. Output file saved as {output_filename}")
    else:
        st.error("Please upload a PDF file")

if __name__ == "__main__":
    main()
