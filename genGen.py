# import streamlit as st
import tempfile
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

os.environ["OPENAI_API_KEY"] = ""


# prompt_template_questions = """
#     你是一個根據提供內容產生中文問題的專家。
#     你的目標是幫助一個使用者了解文件內容。
#     你可以通過提問以下內容來幫助使用者了解文件內容：

#     ------------
#     {text}
#     ------------

#     創建中文問題，確保不會丟失任何重要信息。
#     QUESTIONS:
#     """
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

# refine_template_questions = """
#     你是一個根據提供內容產生問題的專家。
#     你的目標是幫助一個使用者了解文件內容。我們已經收到一些練習問題，但還不夠全面: {existing_answer}.
#     我們可以選擇完善現有的問題或添加新的問題。
#     (只有在必要時)附上一些更多的上下文。

#     ------------
#     {text}
#     ------------

#     根據提供的上下文，用中文修改原始問題，如果提供的上下文沒有幫助，保持原始問題。          
#     QUESTIONS:
#     """
refine_template_questions = """
    You are an expert in generating questions based on provided content.
    Your goal is to help a user understand the document content. We have received some practice questions, but they are not comprehensive enough: {existing_answer}.
    We can choose to improve the existing questions or add new ones.
    (Only if necessary) include some additional context.
    If the existing questions are helpful for understanding the text, keep them unchanged.
    Ensure the total number of questions is at least {num_of_question}.

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


file_path = "WWF_Plastic_Policy_Summit_2024_Key_Takeaways_-1-6.pdf"
loader = PyPDFLoader(file_path)
data = loader.load()
text_question_gen = ''
for page in data:
    text_question_gen += page.page_content
text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split text into chunks for question generation
text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)
docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]
print(len(docs_question_gen))
num_of_question = int(len(docs_question_gen)/3)


llm_question_gen = ChatOpenAI(
        temperature = 0.3,
        model = "gpt-3.5-turbo"
    )
question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                              question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
# questions = question_gen_chain.run(docs_question_gen)
questions = question_gen_chain.invoke({'input_documents':docs_question_gen, 'num_of_question':num_of_question})
llm_answer_gen = ChatOpenAI(
        temperature = 0.3,
        model = "gpt-3.5-turbo"
    )
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs_question_gen, embeddings)
answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff",
                                                   retriever=vector_store.as_retriever())
question_text = questions['output_text']
question_list = questions['output_text'].split("\n")


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
for question in question_list:
    complex_question = complexify_chain.invoke(question)
    complex_questions.append(complex_question['text'])


with open("output.txt", "w") as file:
        for question in complex_questions:
                print("Question: ", question)
                file.write(question + "\n")
                # answer = answer_gen_chain.run(question)
                # print("Answer: ", answer)
                print("--------------------------------------------------\n\n")
print("已成功輸出成txt文件")