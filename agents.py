from crewai import Agent

class ProjectAgents():
    
    def Problem_Definition_Agent(self, model):
        return Agent(
            role='Problem_Definition_Agent',
            goal="""clarify the machine learning problem the user wants to solve, 
                identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
            backstory="""You are an expert in understanding and defining machine learning problems. 
                Your goal is to extract a clear, concise problem statement from the user's input, 
                ensuring the project starts with a solid foundation.""",
            verbose=True,
            allow_delegation=False,
            llm=model,
        )
        
    def Data_Assessment_Agent(self, model):
        return Agent(
            role='Data_Assessment_Agent',
            goal="""evaluate the data provided by the user, assessing its quality, 
                suitability for the problem, and suggesting preprocessing steps if necessary.""",
            backstory="""You specialize in data evaluation and preprocessing. 
                Your task is to guide the user in preparing their dataset for the machine learning model, 
                including suggestions for data cleaning and augmentation.""",
            verbose=True,
            allow_delegation=False,
            llm=model,
        )
            
    def Model_Recommendation_Agent(self, model):
        return Agent(
            role='Model_Recommendation_Agent',
            goal="""suggest the most suitable machine learning models based on the problem definition 
                and data assessment, providing reasons for each recommendation.""",
            backstory="""As an expert in machine learning algorithms, you recommend models that best fit 
                the user's problem and data. You provide insights into why certain models may be more effective than others,
                considering classification vs regression and supervised vs unsupervised frameworks.""",
            verbose=True,
            allow_delegation=False,
            llm=model,
        ) 
        
    def Starter_Code_Generator_Agent(self, model):
        return Agent(
            role='Starter_Code_Generator_Agent',
            goal="""generate starter Python code for the project, including data loading, 
                model definition, and a basic training loop, based on findings from the problem definitions,
                data assessment and model recommendation""",
            backstory="""You are a code wizard, able to generate starter code templates that users 
                can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
            verbose=True,
            allow_delegation=False,
            llm=model
        )
        
    def Summarization_Agent(self, model):
        return Agent(
            role='Starter_Code_Generator_Agent',
            goal="""Summarize findings from each of the previous steps of the ML discovery process.
                Include all findings from the problem definitions, data assessment and model recommendation 
                and all code provided from the starter code generator.
                """,
            backstory="""You are a seasoned data scientist, able to break down machine learning problems for
                less experienced practitioners, provide valuable insight into the problem and why certain ML models
                are appropriate, and write good, simple code to help get started on solving the problem.
                """,
            verbose=True,
            allow_delegation=False,
            llm=model,
        )