
# import streamlit as st
# # Streamlit Page Config
# st.set_page_config(page_title="Growvy Chatbot", page_icon="🤖", layout="centered")
# import os 
# import logging 
# from dotenv import load_dotenv
# from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
# from langchain_openai import ChatOpenAI    
# from langchain_core.vectorstores import VectorStore 
# from langchain_pinecone import PineconeVectorStore 
# from langchain_community.embeddings import OpenAIEmbeddings    
# from langchain.chains import create_retrieval_chain 
# from langchain.chains.combine_documents import create_stuff_documents_chain 
# from langchain.prompts import PromptTemplate    
# from langchain.text_splitter import RecursiveCharacterTextSplitter 
# from langchain.schema import Document 
# from pinecone import Pinecone, ServerlessSpec 
# from pydantic import SecretStr
# from langchain.globals import set_verbose

# set_verbose(True)




# # Load env
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# # Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()


# # Pinecone setup
# pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
#     pinecone_client.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,
#         metric='cosine',
#         spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENVIRONMENT)
#     )

# index = pinecone_client.Index(PINECONE_INDEX_NAME)

# # Embeddings
# # # Initialize embedding model
# embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_type=OPENAI_API_KEY)
# vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# # Load website content from text file
# if os.path.exists("website_content.txt"):
#     with open("website_content.txt", "r", encoding="utf-8") as f:
#         website_text = f.read()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.create_documents([website_text])
#     vector_store.add_documents(docs)
#     logger.info("✅ Website content loaded and indexed.")
# else:
#     st.error("❌ website_content.txt not found!")

# # Chat LLM
# # # LLM setup
# llm_chat = ChatOpenAI(
#     temperature=0.9, 
#     model='gpt-4o-mini', 
#     api_key=SecretStr(OPENAI_API_KEY)
# )
# import base64

# # Load image and encode to base64
# def get_base64_image(image_path):
#     with open(image_path, "rb") as image_file:
#         encoded = base64.b64encode(image_file.read()).decode()
#     return encoded

# # Convert your PNG to base64
# image_base64 = get_base64_image("growvy_logo.png")
# # Display Growvy logo + assistant title
# # st.markdown(f"""
# #         <div style='text-align: center;'>
# #             <img src='data:image/png;base64,{image_base64}' width='100' style='margin-bottom: 10px;' />
# #             <h2 style='margin: 0; font-size: 24px; color: #2E7D32;'>Growvy AI Assistant</h2>
# #             <p style='color: #555; font-size: 14px;'>Ask anything about Growvy's services, features, jobs, or pricing.</p>
# #         </div>
# #     """, unsafe_allow_html=True)
# # Prompt
# qa_prompt_template = """
# You are a helpful assistant answering based on the context below.

# Context:
# {context}

# Question: {input}
# """
# qa_prompt = PromptTemplate.from_template(qa_prompt_template)
# qa_chain = create_stuff_documents_chain(llm_chat, qa_prompt)
# retrieval_chain = create_retrieval_chain(
#     retriever=vector_store.as_retriever(),
#     combine_docs_chain=qa_chain
# )

# # Chat History
# if "history" not in st.session_state:
#     st.session_state.history = []



# # Display chat
# for msg in st.session_state.history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # User input
# query = st.chat_input("💬 Ask a question about Growvy...")
# if query:
#     st.chat_message("user").markdown(query)
#     st.session_state.history.append({"role": "user", "content": query})

#     with st.spinner("Thinking..."):
#         try:
#             response = retrieval_chain.invoke({"input": query})
#             answer = response["answer"]
#         except Exception as e:
#             logger.error(f"Error: {e}")
#             answer = "Sorry, I couldn't find an answer. Please rephrase."

#     st.chat_message("assistant").markdown(answer)
#     st.session_state.history.append({"role": "assistant", "content": answer})
import streamlit as st
import os
import logging
import base64
from dotenv import load_dotenv
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pydantic import SecretStr
from langchain.globals import set_verbose

# Streamlit Config
st.set_page_config(page_title="Growvy Chatbot", page_icon="🤖", layout="centered")
set_verbose(True)
load_dotenv()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Pinecone setup
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in [index.name for index in pinecone_client.list_indexes()]:
    pinecone_client.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region=PINECONE_API_ENVIRONMENT)
    )

index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Embeddings & Vector Store
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_type=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

# Load website content
if os.path.exists("website_content.txt"):
    with open("website_content.txt", "r", encoding="utf-8") as f:
        website_text = f.read()

    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).create_documents([website_text])
    vector_store.add_documents(docs)
    logger.info("✅ Website content loaded and indexed.")
else:
    st.error("❌ website_content.txt not found!")

# LLM Setup
llm_chat = ChatOpenAI(
    temperature=0.9,
    model='gpt-4o-mini',
    api_key=SecretStr(OPENAI_API_KEY)
)

import base64

# Load image and encode to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert your PNG to base64
image_base64 = get_base64_image("growvy_logo.png")
# Display Growvy logo + assistant title
st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{image_base64}' width='100' style='margin-bottom: 10px;' />
            <h2 style='margin: 0; font-size: 24px; color: #2E7D32;'>Growvy AI Assistant</h2>
            <p style='color: #555; font-size: 14px;'>Ask anything about Growvy's services, features, jobs, or pricing.</p>
        </div>
    """, unsafe_allow_html=True)

qa_prompt_template = """
You are a smart and helpful AI assistant for the Growvy website. 
Always try to provide an informative and helpful answer, even if the exact details are not present in the context below. 
Use general knowledge and logical reasoning to respond in a useful way. 
Never say the information is not available or not found — instead, guide the user helpfully.

Context:
{context}

Question:
{input}

Helpful Answer:
"""
qa_prompt = PromptTemplate.from_template(qa_prompt_template)
qa_chain = create_stuff_documents_chain(llm_chat, qa_prompt)
retrieval_chain = create_retrieval_chain(
    retriever=vector_store.as_retriever(),
    combine_docs_chain=qa_chain
)




# Chat Session
if "history" not in st.session_state:
    st.session_state.history = []

# Display Chat Messages
for msg in st.session_state.history:
    align = "flex-start" if msg["role"] == "assistant" else "flex-end"
    bg_color = "#F0F0F0" if msg["role"] == "assistant" else "#DCF8C6"
    border_radius = "0.5rem"
    st.markdown(f"""
        <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
            <div style="background-color: {bg_color}; padding: 10px 15px; max-width: 80%; border-radius: {border_radius}; box-shadow: 0px 1px 5px rgba(0,0,0,0.1);">
                {msg['content']}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Chat Input
query = st.chat_input("💬 Ask a question about Growvy...")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #DCF8C6; padding: 10px 15px; max-width: 80%; border-radius: 0.5rem; box-shadow: 0px 1px 5px rgba(0,0,0,0.1);">
                {query}
            </div>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        try:
            response = retrieval_chain.invoke({"input": query})
            answer = response["answer"]
        except Exception as e:
            logger.error(f"Error: {e}")
            answer = "⚠️ Sorry, I couldn't find an answer. Please try again."

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #F0F0F0; padding: 10px 15px; max-width: 80%; border-radius: 0.5rem; box-shadow: 0px 1px 5px rgba(0,0,0,0.1);">
                {answer}
            </div>
        </div>
    """, unsafe_allow_html=True)
