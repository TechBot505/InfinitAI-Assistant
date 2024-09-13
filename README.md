# InfinitAI Machine Learning Assistant
  The InfinitAI Machine Learning Assistant is a Streamlit-based application designed to guide users through the process of defining, assessing, and solving machine learning problems. Leveraging the   CrewAI library, this application utilizes a team of AI agents to provide assistance at various stages of the machine learning workflow.
  
  Deployed Link: [InfinitAI](https://crewai-assistant-techbot-505.streamlit.app/)

### Overview
  The InfinitAI Machine Learning Assistant simplifies the complex process of working on machine learning projects by breaking it down into manageable steps. It incorporates AI agents specialized in   different tasks such as problem definition, data assessment, model recommendation, and starter code generation. Users interact with the assistant through a Streamlit interface, providing input on   their ML problem and selecting which steps of the workflow they'd like to include.

### Features
  * **Problem Definition**: Users can describe their machine learning problem, and the assistant provides guidance on refining the problem statement.
  * **Data Assesment**: Users have the option to upload a sample CSV file of their data for assessment. The assistant evaluates the data to provide insights and recommendations.
  * **Model Recommendation**: Based on the problem description and data assessment, the assistant suggests suitable machine learning models to explore further.
  * **Starter Code Generation**: The assistant generates starter Python code tailored to the defined problem and recommended model.

### Getting Started
  #### Installation
  To run the InfinitAI Machine Learning Assistant locally, follow these steps:
  1. Clone this repository to your local machine: `git clone https://github.com/your-username/infinitai-ml-assistant.git`
  2. Navigate to the project directory: `cd infinitai-ml-assistant`
  3. Install the required dependencies using pip: `pip install -r requirements.txt`

  #### Running the App
  Once the dependencies are installed, you can run the application using Streamlit.
    `streamlit run app.py`

### Usage
1. Upon launching the application, you'll be presented with a Streamlit interface.
2. Describe your machine learning problem in the text area provided.
3. Select the steps you'd like to include in the workflow by checking the corresponding checkboxes.
4. If you have sample data, you can upload a CSV file using the file uploader.
5. Click the "Submit" button to initiate the workflow.
6. The assistant will guide you through the selected steps, providing insights and recommendations along the way.
7. The final output will be displayed in the chat interface, summarizing the results of the workflow.

### Customization
  * **Model Selection**: Users can customize the machine learning model used by selecting from available options in the sidebar.
  * **Model Temprature**: Adjust the temperature parameter to control the randomness of the model's output.
