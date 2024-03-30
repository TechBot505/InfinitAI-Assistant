import streamlit as st
import pandas as pd
import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def main():
    
    st.title("CrewAI Machine Learning Assistant")
    multiline_text = """
    The CrewAI Machine Learning Assistant is designed to guide users through the process of defining, assessing, and solving machine learning problems. It leverages a team of AI agents, each with a specific role, to clarify the problem, evaluate the data, recommend suitable models, and generate starter Python code. Whether you're a seasoned data scientist or a beginner, this application provides valuable insights and a head start in your machine learning projects.
    """
    st.markdown(multiline_text, unsafe_allow_html=True)
    
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    

if __name__ == '__main__':
    main()

