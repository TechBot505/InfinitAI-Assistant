from crewai import Task

class ProjectTasks():
    
    def task_define_problem(self, agent, user_question):
        return Task(
            description="""Clarify and define the machine learning problem, 
                including identifying the problem type and specific requirements.
                
                Here is the user's problem:

                {ml_problem}
                """.format(ml_problem=user_question),
            agent=agent,
            expected_output="A clear and concise definition of the machine learning problem."
        )
        
    def task_assess_data_1(self, agent, df, uploaded_file):
        return Task(
            description="""Evaluate the user's data for quality and suitability, 
                suggesting preprocessing or augmentation steps if needed.
                
                Here is a sample of the user's data:

                {df}

                The file name is called {uploaded_file}
                
                """.format(df=df.head(),uploaded_file=uploaded_file),
            agent=agent,
            expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
        )
        
    def task_access_data_2(self, agent):
        return Task(
            description="""The user has not uploaded any specific data for this problem,
                but please go ahead and consider a hypothetical dataset that might be useful
                for their machine learning problem. 
                """,
            agent=agent,
            expected_output="A hypothetical dataset that might be useful for the user's machine learning problem, along with any necessary preprocessing steps."
        )
    
    def task_recommend_model(self, agent):
        return Task(
            description="""Suggest suitable machine learning models for the defined problem 
            and assessed data, providing rationale for each suggestion.""",
            agent=agent,
            expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
        )
        
    def task_generate_code(self, agent):
        return Task(
            description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), 
            including snippets for package import, data handling, model definition, and training
            """,
            agent=agent,
            expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        )