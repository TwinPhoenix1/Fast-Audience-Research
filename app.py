from functions import *
from typing import List
from pydantic import BaseModel
from llama_cpp.llama import Llama
import pandas as pd

corpus = scrape_subreddit(
    client_id = "",
    client_secret = "",
    username = "", 
    password = "",
    user_agent = "Fast Audience Research (FAR) by u/TwinPhoenix1",
    subreddit = 'BaseBuildingGames',
    limit = 12)

llm = Llama(
    model_path = "D:\GGUF\Hermes-2-Pro-Mistral-7B.Q8_0.gguf", 
    chat_format = "chatml", 
    n_ctx = 8192, 
    n_gpu_layers = 0)

goals_num = 5 
do_not_want_num = 5
wants_num = 5

class ThreadData(BaseModel):
    Goals: str
    Frustrations: List[str]
    Satisfaction: List[str]

class SummarizedData(BaseModel):
    Goals: List[str]
    Frustrations: List[str]
    Satisfaction: List[str]

schema = json.dumps(ThreadData.model_json_schema())
schema_summarize = json.dumps(SummarizedData.model_json_schema())

prompt_start = """<|im_start|>system
            I have collected data from a Reddit thread in an industry-specific subreddit. My intention 
            is to perform market research, and the users in this subreddit are part of my audience. 
            
            Please analyze the titles, original posts, and comments of the thread, and extract data 
            for each poster and commenter where relevant data is present:

            1. What goal (or goals) are they trying to achieve that are relevant to the product or service?
            2. What are their frustrations with achieving the goal?
            3. What has worked for them and what is their experience with it? 

            The data will be provided in JSON format.

            For your output, here's the json schema you must adhere to: \n<schema>\n{schema}\n<schema>

            It is critical that you respond with a single JSON string where you provide all the requested information. 
            Don't include any other verbose explanations and don't include markdown syntax anywhere.
            Use full, descriptive phrases to describe each goal, frustration and satisfaction.
            
            What follows is the data for you to analyze:
            """

prompt_start = prompt_start.replace("{schema}", schema)

prompt_sentiment_summarization = """<|im_start|>system
            I have collected data from Reddit threads in the r/BaseBuildingGames subreddit. 
            My intention is to perform market research, and this subreddit is representative of my 
            audience. Using NLP, I extracted summarized data about the goals, frustrations and reasons for satisfaction
            for each poster and commenter in the threads I scraped. 

            You will receive the data in JSON format.

            Summarize the list down to the """ + str(goals_num) + """ main goals 
            people have. For each goal, provide a ranked list of the """ + str(do_not_want_num) + """ most 
            common reasons for frustration and the """ + str(wants_num) + """ most common reasons for
            satisfaction. Avoid saying just the game title as a reason for frustration or satisfaction - 
            go in detail about the actual reason. Minimize the overlap between the reasons you list, 
            so that your list can help me understand the posters in greater detail. 
            
            Produce your response in the following format, while making sure your output contains 
            the correct number of goals, frustrations and reasons for satisfaction that I requested.
            
            For your output, here's the json schema you must adhere to: \n<schema>\n{schema_summarize}\n<schema>
            
            It is very critical that you respond in a single JSON string. Don't include any other verbose explanations 
            and don't include markdown syntax anywhere.

            What follows is the data for you to analyze:"""

prompt_sentiment_summarization = prompt_sentiment_summarization.replace("{schema_summarize}", schema_summarize)

analysis_by_thread = get_analysis(corpus, prompt_start, llm)

final_analysis = refine_analysis(analysis_by_thread, 
                                 prompt_sentiment_summarization, 
                                 llm)

df = pd.read_json(final_analysis)
df.to_csv('final_analysis.csv', encoding='utf-8', index=False)
