from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
import streamlit as st
from streamlit_chat import message
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from operator import itemgetter
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
                                                          ForgotError,
                                                          LoginError,
                                                          RegisterError,
                                                          ResetError,
                                                          UpdateError)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import boto3
import getpass
import os

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs=dict(temperature=0.9),streaming = True
)

loader = PyPDFLoader("database/contact.pdf")
data = loader.load()
data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=200)
splits = data_split.split_documents(data)
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()


# Prompt templates
_template = """ Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You're an AI expert in psychology and psychiatry named Anastasia created by Daffa Tsabit Murtadha.
You have several task are to give mental health and personal development advice to users and become their mental support. 
You will be confused if the user ask about something out of topic. 
Only give the user about the professional data when only they ask. 
Be smart and creative. don't throw the same answer in every response.
Your response's language based on user language.  
{context}
</context> """
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def _format_chat_history(chat_history):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

class ChatHistory(BaseModel):
    chat_history: list = Field(..., extra={"widget": {"type":"chat"}})
    question: str
    
# Runnable parts
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        ) | CONDENSE_QUESTION_PROMPT | llm | StrOutputParser(),
    ),
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel({
    "question": lambda x: x["question"],
    "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    "context": _search_query | retriever | _combine_documents,
}).with_types(input_type=ChatHistory)

class StreamingChain:
    def __init__(self, chain):
        self.chain = chain

    def stream(self, input_data):
        for result in self.chain.stream(input_data):
            yield result

# Chaining
streaming_chain = StreamingChain(_inputs | ANSWER_PROMPT | llm | StrOutputParser())

st.set_page_config(page_title='Start Consulting', layout='wide', page_icon="./fav.ico")

# Initialize session state for chat history if not already exists
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    st.button("Clear Chat History", on_click=lambda: st.session_state.chat_history.clear())
    st.title("Chat with Anastasia")
    st.write("   Hai! Saya Anastasia. Apa yang bisa saya bantu?")
    with st.sidebar:
        st.write(f'Welcome *{st.session_state["name"]}*')
        authenticator.logout()
        st.divider()
        st.image('./fav.ico', width=100, use_column_width= "never")
        st.write("## Anastasia.ai")
        st.caption("Personal AI Assistant in Mental Health and Personal Development")
        st.caption("Several use cases: ")
        st.caption("1. Consultation")
        st.caption("2. Discussion")
        st.caption("3. Ask for medical check-up with real mental health professional")
    for user_msg, ai_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(ai_msg)

    # Input box for user question
    question = st.chat_input("Your question:")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            user_msg_placeholder = st.empty()
            user_msg_placeholder.markdown(question)
    
        chat_obj = ChatHistory(chat_history=st.session_state.chat_history, question=question)
    
        # Show AI response message container and temporarily hold full response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
    
            # Stream the response from the chain
            for chunk in streaming_chain.stream(chat_obj.dict()):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")  # Temporary indicator for streaming
    
            response_placeholder.markdown(full_response)  # Update to final response
            # Add messages to chat history
            st.session_state.chat_history.append((question, full_response))

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')