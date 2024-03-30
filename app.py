import streamlit as st
import pandas as pd
import os
from crewai import Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tasks import ProjectTasks
from agents import ProjectAgents

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
    
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model,
        temperature=0
    )
    
    tasks = ProjectTasks()
    agents = ProjectAgents()
    
    Problem_Definition_Agent = agents.Problem_Definition_Agent(model=llm)
    Data_Assessment_Agent = agents.Data_Assessment_Agent(model=llm)
    Model_Recommendation_Agent = agents.Model_Recommendation_Agent(model=llm)
    Starter_Code_Generator_Agent = agents.Starter_Code_Generator_Agent(model=llm)
    
    user_question = st.text_area("Describe your ML problem:")
    submit_button = st.button("Submit")
    data_upload = False
    uploaded_file = st.file_uploader("Upload a sample .csv of your data (optional)")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file).head(5)
            data_upload = True
            st.write("Data successfully uploaded and read as DataFrame:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            
    if submit_button and user_question:
        
        task_define_problem = tasks.task_define_problem(agent=Problem_Definition_Agent, user_question=user_question)
        
        if data_upload:
            task_assess_data = tasks.task_assess_data_1(agent=Data_Assessment_Agent, df=df, uploaded_file=uploaded_file.name)
        else:
            task_assess_data = tasks.task_access_data_2(agent=Data_Assessment_Agent)
            
        task_recommend_model = tasks.task_recommend_model(agent=Model_Recommendation_Agent)
        
        task_generate_code = tasks.task_generate_code(agent=Starter_Code_Generator_Agent)
        
        crew = Crew(
            agents=[Problem_Definition_Agent, Data_Assessment_Agent, Model_Recommendation_Agent,  Starter_Code_Generator_Agent],
            tasks=[task_define_problem, task_assess_data, task_recommend_model,  task_generate_code],
            verbose=2
        )
        
        result = crew.kickoff()
        
        st.write(result)
    

if __name__ == '__main__':
    main()

