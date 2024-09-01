from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain
import os
import streamlit as st
import time

def main(file_path, progress_bar):
    os.environ["OPENAI_API_KEY"] = ""

    prompt_template_questions = """
        You are an expert in generating English questions based on provided content.
        Your goal is to help a user understand the document content.
        You can help the user understand the document content by asking questions about the following content:

        ------------
        {text}
        ------------

        Create English questions, ensuring that no important information is missed.
        QUESTIONS:
        """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

    refine_template_questions = """
        You are an expert in generating questions based on provided content.
        Your goal is to help a user understand the document content. We have received some practice questions, but they are not comprehensive enough: {existing_answer}.
        We can choose to improve the existing questions or add new ones.
        (Only if necessary) include some additional context.
        If the existing questions are helpful for understanding the text, keep them unchanged.
        Please make sure to keep the number of generated questions to  {num_of_question}; this is very important.
        Please make sure to keep the number of generated questions to  {num_of_question}; this is very important.
        Please make sure to keep the number of generated questions to  {num_of_question}; this is very important.

        ------------
        {text}
        ------------

        Based on the provided context, modify the original questions in English. If the provided context is not helpful, keep the original questions unchanged.
        QUESTIONS:
        """
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "num_of_question", "text"],
        template=refine_template_questions,
    )

    docs_question_gen, num_of_question = chunk_text(file_path)
    question_list = gen_questions(docs_question_gen, num_of_question, PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS, progress_bar)
    complex_questions_list = complex_questions(question_list, progress_bar)
    output_questions(complex_questions_list)
    
def chunk_text(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_question_gen = ''
    for page in data:
        text_question_gen += page.page_content
    text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split text into chunks for question generation
    text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)
    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]
    print(f"number of chunks: {len(docs_question_gen)}")
    num_of_question = int(len(docs_question_gen)/3)
    
    return docs_question_gen, num_of_question

def gen_questions(docs_question_gen, num_of_question, PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS, progress_bar):
    with st.spinner('Generating questions first time..., please wait for a while...'):
        llm_question_gen = ChatOpenAI(
                temperature = 0.3,
                model = "gpt-3.5-turbo"
            )
        question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                                    question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

        questions = question_gen_chain.invoke({'input_documents':docs_question_gen, 'num_of_question':num_of_question})
        question_list = questions['output_text'].split("\n")
        
    return question_list

def complex_questions(question_list, progress_bar):
    complexify_prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""You are now a student who is good at asking questions.
                    I will give you a question, and please help me ask it in a more complex way.
                    The question is as follows: {question}.
                    Remember to keep the question number."""
    )

    llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")

    complexify_chain = LLMChain(
        llm=llm,
        prompt=complexify_prompt_template
    )

    complex_questions = []
    progress_text = "Generating questions second time..., please wait for a while..."
    for i, question in enumerate(question_list, start=1):
        progress_bar.progress(i / len(question_list), text=progress_text)
        complex_question = complexify_chain.invoke(question)
        complex_questions.append(complex_question['text'])
    
    return complex_questions    

def output_questions(complex_questions_list):
    with open("Q_output.txt", "w") as file:
        for question in complex_questions_list:
            file.write(question + "\n")

if __name__ == "__main__":
    main()