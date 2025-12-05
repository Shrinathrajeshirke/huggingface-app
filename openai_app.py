## import required libraries

import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

## loads all env variables
load_dotenv()

## langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A chatbot with OPENAI"


## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. please response to the user queries"),
        ("user","question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatOpenAI(model=llm, openai_api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

## title of the app
st.title("Enhanced Q&A chatbot with OpenAI")

## sidebar for settings
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your OPEN AI API Key:", type="password")

## drop down to select various open AI models
llm = st.sidebar.selectbox("select an Open AI model",["gpt-4o", "gpt-4-turbo","gpt-4"])

## adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max tokens", min_value=50, max_value=300, value=150)

## main interface for user input
st.write("Go ahead and ask a question")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("please provide the query")