from llama_cpp.llama import Llama, LlamaGrammar
import requests
import time 
import praw 
import re
import json

# set to -1 if your GPU has 10+ GB VRAM, set to whatever else if you know what you're doing:
n_gpu_layers = 0

def scrape_subreddit(client_id, client_secret, username, password, user_agent, subreddits, limit):
    
    client_auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    post_data = {
        "grant_type": "password", 
        "username": username, 
        "password": password}
    headers = {"User-Agent": user_agent}

    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
    response.json()

    reddit = praw.Reddit(
        client_id = client_id,
        client_secret = client_secret,
        password = password,
        user_agent = user_agent,
        username = username,
    )

    subreddit = reddit.subreddit(subreddit)

    top_threads = subreddit.hot(limit=limit)
    threads = [thread for thread in top_threads 
            if len(re.findall(r'\w+', thread.selftext)) > 5 
            and thread.num_comments > 4] 

    print("Threads fetched")
    corpus = []
    i=0

    for thread in threads:
        title = thread.title
        content = thread.selftext
        comments = [comment.body for comment in thread.comments 
                if len(re.findall(r'\w+', comment.body)) > 5]
        print(i)
        i+=1
        thread_content = {
            'Title': title,
            'Content': content,
            'Comments': comments}
        corpus.append(thread_content)
        time.sleep(1)
    return corpus


def get_query(corpus, prompt_start):
    for i in range(0, len(corpus)):
        query_candidate = prompt_start + str(corpus[0:i])
        if len(re.findall(r'\w+', query_candidate)) > 5500:
            return (prompt_start + str(corpus[0:i-1]) + " <|im_end|>"), corpus[i-1:]
    return (prompt_start + str(corpus) + " <|im_end|>"), list()

def get_output(prompt, llm, grammar):
    response = llm(prompt, grammar=grammar, max_tokens=8192)
    response = response['choices'][0]['text'].replace("{\n    ", "{")
    response = response.replace("[{\\", "")
    response = json.loads(response.replace("\n   ", ""))
    return response
    
def get_analysis(corpus, prompt_start, llm):
    analysis = []
    grammar = LlamaGrammar.from_file("thread_schema.gbnf")
    while len(corpus) > 0:
        prompt = prompt_start + str(corpus.pop()) + " <|im_end|>"
        while True:
            try:
                output = get_output(prompt, llm, grammar)
                analysis.append(output)
            except: 
                continue
            else:
                break
    return analysis

def refine_analysis(analysis_by_thread, prompt_sentiment_summarization, llm):
    prompt = prompt_sentiment_summarization + "\n\n"+ json.dumps(analysis_by_thread) + " <|im_end|>"
    grammar = LlamaGrammar.from_file("summary_schema.gbnf")
    output = get_output(prompt, llm, grammar)
    return output

