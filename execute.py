#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## load ChatQA-1.5 tokenizer and model
model_id = "C:/Users/jywun/Desktop/NYCU/模組/QA_ARAG/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

#%%
embedding_model_path = "BAAI/bge-large-en-v1.5"
messages = [
    {"role": "user", "content": ""}
]
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
low_confidence_first = []
low_confidence_second = []
first_answer = []
second_answer = [] 
max_length = 512

#%%
def get_reference_document(path, chunk_size = 500, chunk_overlap = 100):
    with pdfplumber.open(path) as pdf: 
        content = ''
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            page_content = '\n'.join(page.extract_text().split('\n')[:-1])
            content = content + page_content
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_text(content)
        docs = [Document(page_content=t) for t in text_chunks]

    return docs

def get_question_from_txt(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    question = []
    for line in lines:
        text_without_numbers_and_periods = re.sub(r'[\d.]+', '', line)
        cleaned_question = text_without_numbers_and_periods.replace('\n', '')
        question.append(cleaned_question)
        
    return question

def replace_message_content(question):
    messages = [
    {"role": "user", "content": ""}
    ]
    messages[0]['content'] = question.strip()

    return messages

def retrival(vector_store, search_kwargs, question):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': search_kwargs})
    doc = retriever.invoke(question)

    result = []
    for part in doc:
            if part.page_content not in result:
                result.append(part.page_content)

    return result

def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."


            ## only apply this instruction for the first user turn
    messages[0]['content'] = instruction + " Question: " + messages[0]['content']


    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\nAssistant:"
    formatted_input = system + "\n\n" + "retrival 1:\n" + context[0] + "\n\n" + conversation
    
    return formatted_input 

def get_formatted_input_cloze(messages, context, generated_tokens):
    system = "System: This is an artificial intelligence assistant that can complete sentences with uncertain parts removed due to low confidence. The assistant provides contextually accurate answers to the user's questions."
    instruction = "Please modify answer based on context and user questions."
    generated_tokens = generated_tokens[1:-1]
    cloze = tokenizer.decode(generated_tokens)
    messages[0]['content'] = instruction + " Question: " + messages[0]['content']

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + " Answer:" + cloze + "\nAssistant:"
    formatted_input = system + "\n\n" + "retrival 1:\n" + context[0]  + "\n" "retrival 2:\n" + context[-1] + "\n\n" + conversation
    
    return formatted_input

def find_below_threshold_indices(generated_probabilities, threshold):
    below_threshold_indices = []
    for i, val in enumerate(generated_probabilities):
        if val < threshold:
            below_threshold_indices.append(i)

    return below_threshold_indices

def remove_below_threshold_tokens(generated_tokens, generated_probabilities, threshold, turn=0):
    below_threshold_indices = find_below_threshold_indices(generated_probabilities, threshold)
    if turn == 1:
        low_confidence_first.append(len(below_threshold_indices))
        for i in reversed(below_threshold_indices):
            generated_tokens.pop(i)
    else:
        low_confidence_second.append(len(below_threshold_indices))
        

    return generated_tokens

def remove_bos_eos(generated_text):
    result = generated_text.replace("<|begin_of_text|> ", "")
    result = result.replace("<|end_of_text|>", "")

    return result

def first_retrieval_answer(messages, vector_store, question, question_num=0):
    retrival_doc = retrival(vector_store, 1, question[question_num])
    formatted_input = get_formatted_input(messages, retrival_doc)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
    tokenized_prompt_recursive = tokenized_prompt.input_ids
    generated_tokens = []
    generated_probabilities = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(tokenized_prompt_recursive)
            logits = outputs.logits

            temperature = 0.7  
            logits = logits / temperature
            probabilities = F.softmax(logits[:, -1, :], dim=-1).squeeze()

            # next_token = torch.argmax(probabilities[:, -1, :], dim=-1).unsqueeze(-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            print(tokenizer.decode(next_token.item()), end="")

            generated_tokens.append(next_token.item())
            generated_probabilities.append(probabilities.squeeze()[next_token])
            # 将生成的标记加入到输入中以生成下一个标记
            tokenized_prompt_recursive = torch.cat([tokenized_prompt_recursive, next_token.unsqueeze(0)], dim=1)
            if next_token.item() in terminators:
                break
    generated_text = tokenizer.decode(generated_tokens)
    first = remove_bos_eos(generated_text)
    first_answer.append(first)
    
    return generated_tokens, generated_probabilities

def second_retrieval_answer(second_answer, messages, vector_store, generated_tokens, question, question_num=0):
    retrival_doc = retrival(vector_store, 2, question[question_num])
    formatted_input = get_formatted_input_cloze(messages, retrival_doc, generated_tokens)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)
    tokenized_prompt_recursive = tokenized_prompt.input_ids
    generated_tokens = []
    generated_probabilities = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(tokenized_prompt_recursive)
            logits = outputs.logits

            temperature = 0.7  
            logits = logits / temperature
            probabilities = F.softmax(logits[:, -1, :], dim=-1).squeeze()
            
            next_token = torch.argmax(probabilities).unsqueeze(-1)
            # next_token = torch.multinomial(probabilities, num_samples=1)
            # print(tokenizer.decode(next_token.item()))

            generated_tokens.append(next_token.item())
            generated_probabilities.append(probabilities.squeeze()[next_token])
            # 将生成的标记加入到输入中以生成下一个标记
            tokenized_prompt_recursive = torch.cat([tokenized_prompt_recursive, next_token.unsqueeze(0)], dim=1)
            if next_token.item() in terminators:
                break
    generated_text = tokenizer.decode(generated_tokens)
    second = remove_bos_eos(generated_text)
    print("finish")
    second_answer.append(second)
    
    return generated_tokens, generated_probabilities, second_answer
