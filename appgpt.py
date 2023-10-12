import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer
# from langchain import PromptTemplate
from langchain.prompts import PromptTemplate

openapi_key = st.secrets["OPENAI_API_KEY"]

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
# embedding_model_name = "all-MiniLM-L6-v2"
# "with" notation


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file")
    st.header("Shahada GPT")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        # uploaded_files = st.file_uploader("Upload your file", type=[
        #                                   'pdf'], accept_multiple_files=False)
        openai_api_key = openapi_key
        # openai_api_key = st.text_input("OpenAI API Key", key=openapi_key , type="password")
        # process = st.button("Process")
        st.sidebar.write(""" <div style="text-align: center"> The chatbot will bring answers only from the book Meaning of Quran' by Noor International which is  is a profound and comprehensive exploration of the Quran.This work meticulously delves into the Quranic verses, unraveling their meanings, historical context, and relevance to contemporary life. 'Meaning of Quran' serves as an invaluable resource for those seeking spiritual insight, knowledge, and a more profound connection to the teachings of the Quran.</div>""", unsafe_allow_html=True)
        # process = st.text_area("This GPT is about the book Meaning of Quran' by Noor International which is  is a profound and comprehensive exploration of the Quran, offering readers a deeper understanding of its divine wisdom and guidance. This work meticulously delves into the Quranic verses, unraveling their meanings, historical context, and relevance to contemporary life. Noor International's approach is characterized by scholarly expertise and a commitment to promoting a more profound comprehension of the Quran's message, making it accessible to both Muslims and those interested in the study of Islamic scriptures. 'Meaning of Quran' serves as an invaluable resource for those seeking spiritual insight, knowledge, and a more profound connection to the teachings of the Quran. ",disabled=True)
    # if process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    # files_text = get_files_text(uploaded_files)
    # st.write("File loaded...")
    # get text chunks
    # text_chunks = get_text_chunks(files_text)
    # st.write("file chunks created...")
    # create vetore stores
    embedding_folder_path = '1_en-translation-of-the-meanings-of-the-quran' + '_' + embedding_model_name
    # print(embedding_folder_path)
    vetorestore = load_vectorstore(embedding_folder_path)
    # st.write("Vectore Store Created...")
    # create conversation chain
    st.session_state.conversation = get_conversation_chain(
        vetorestore, openai_api_key)  # for openAI

    st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

# Function to get the input file and read the text from it.


def get_files_text(uploaded_files):
    text = ""
    # for uploaded_file in uploaded_files:
    split_tup = os.path.splitext(uploaded_files.name)
    file_extension = split_tup[1]
    if file_extension == ".pdf":
        text += get_pdf_text(uploaded_files)
        # elif file_extension == ".docx":
        #     text += get_docx_text(uploaded_file)
        # else:
        #     text += get_csv_text(uploaded_file)
    return text

# Function to read PDF Files


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# def get_docx_text(file):
#     doc = docx.Document(file)
#     allText = []
#     for docpara in doc.paragraphs:
#         allText.append(docpara.text)
#     text = ' '.join(allText)
#     return text


# def get_csv_text(file):
#     return "a"

def get_prompt():
    prompt_template = """
        you are an Islamic scholar. As an islamic scholar, you are expert in extracting information related to islamic queries. You are trained to analyze the given paragraphs and extract relevant information from the query.
        If you able to find any relevant information, then you must return it without adding extra information in the paragraphs.
        paragraphs: {context}
        query: {question}
        Relevant information: 
    """
    # prompt_template = """you are an Islamic scholar. If the Answer is not found in the {context}, then return "I don't know", otherwise return the Answer. Don't (add up) anything in that answer. Response must only contain that answer which in the given {context}.Do not go outside the {context}.Specify the answer with in the given {context}. If you don't find anythin related in context then respond with i don't know'
    # Context: {context}
    # Question: {question}"""
    mprompt_url = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"], validate_template=False)
    # chain_type_kwargs = {"prompt": mprompt_url}
    return mprompt_url


def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Using the hugging face embedding models
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_local"):
        knowledge_base = FAISS.load_local("faiss_local", embeddings)
    else:
        knowledge_base = FAISS.from_texts(text_chunks, embeddings)
        knowledge_base.save_local("faiss_local")

    return knowledge_base

    # creating the Vectore Store using Facebook AI Semantic search


def get_conversation_chain(vetorestore, openai_api_key):
    # llm = ChatOpenAI(openai_api_key=openai_api_key,
    #                  model_name='gpt-3.5-turbo-16k', temperature=0)
    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)
    chain_type_kwargs = {"prompt": get_prompt()}
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), chain_type="stuff", retriever = vetorestore.as_retriever(search_type="similarity", search_kwargs={'k': 5}),
                                           chain_type_kwargs=chain_type_kwargs, return_source_documents=False)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vetorestore.as_retriever(),
    #     memory=memory
    # )
    return qa_chain

def load_vectorstore(folder_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")
    if os.path.exists(folder_path):
        vectordb = FAISS.load_local(folder_path,embeddings)
        return vectordb
    else:
        print("Path Not Found")
        
    

def handel_userinput(user_question):
    # user_question += ". Specify the answer with in the given {context}. If you don't find anythin related in context then respond with i don't know"
    # print(user_question)
    with get_openai_callback() as cb:
        result = st.session_state.conversation({'query': user_question})
        response = result['result']
        # source = result['source_documents'][0].metadata['source']
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{response} ")
    # try:
    #     st.session_state.chat_history = response['chat_history']
    # except Exception as e:
    #     pass
    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            # print(messages)
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))


if __name__ == '__main__':
    main()
