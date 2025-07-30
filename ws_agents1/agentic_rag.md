```python
import os
os.chdir("/home/kantundpeterpan/projects/zoomcamp/zcllm/ws_agents1")
```


```python
import json, requests, dotenv
from openai import OpenAI
try:
    if dotenv.load_dotenv(): 
        OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
except KeyError:
    raise ValueError("set api key")
# You can use any model that supports tool calling
MODEL = "google/gemini-2.5-flash"
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)
```


```python
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```


```python
from minsearch import AppendableIndex

index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)
```




    <minsearch.append.AppendableIndex at 0x7f691d5eecf0>




```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results
```


```python
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()
```


```python
def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
```


```python
def llm(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

```


```python
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer
```


```python
print(rag('Can i join after the course started ?'))
```

    Yes, you can still join the course even after the start date. You are eligible to submit homeworks even if you haven't registered. However, be aware of the deadlines for final projects to avoid last-minute rush.


# Agentic RAG


```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT> 
{context}
</CONTEXT>

If CONTEXT is EMPTY, you can use our FAQ database IF you deem the question related to one of the courses
you are responsible for: Machine Learning, MLOps, Data Engineering and LLM Applications.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}
""".strip()
```


```python
question = 'Can I still join the course ?'
context = "EMPTY"
```


```python
prompt = prompt_template.format(
    question = question,
    context = context
)
```


```python
answer_json = llm(prompt)
```


```python
import json
```


```python
answer = json.loads(answer_json)
```


```python
answer['action']
```




    'SEARCH'




```python
def build_context(search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"

    return context.strip()
```


```python
search_results = search(question)
context = build_context(search_results)
prompt = prompt_template.format(question=question, context=context)
print(prompt)
```

    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    At the beginning the context is EMPTY.
    
    <QUESTION>
    Can I still join the course ?
    </QUESTION>
    
    <CONTEXT> 
    section: General course-related questions
    question: Course - Can I still join the course after the start date?
    answer: Yes, even if you don't register, you're still eligible to submit the homeworks.
    Be aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.
    
    section: General course-related questions
    question: Certificate - Can I follow the course in a self-paced mode and get a certificate?
    answer: No, you can only get a certificate if you finish the course with a “live” cohort. We don't award certificates for the self-paced mode. The reason is you need to peer-review capstone(s) after submitting a project. You can only peer-review projects at the time the course is running.
    
    section: General course-related questions
    question: Course - Can I follow the course after it finishes?
    answer: Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.
    You can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.
    
    section: General course-related questions
    question: Course - When will the course start?
    answer: The purpose of this document is to capture frequently asked technical questions
    The exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1
    Subscribe to course public Google Calendar (it works from Desktop only).
    Register before the course starts using this link.
    Join the course Telegram channel with announcements.
    Don’t forget to register in DataTalks.Club's Slack and join the channel.
    
    section: General course-related questions
    question: Course - I have registered for the Data Engineering Bootcamp. When can I expect to receive the confirmation email?
    answer: You don't need it. You're accepted. You can also just start learning and submitting homework without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.
    </CONTEXT>
    
    If CONTEXT is EMPTY, you can use our FAQ database IF you deem the question related to one of the courses
    you are responsible for: Machine Learning, MLOps, Data Engineering and LLM Applications.
    In this case, use the following output template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>"
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }



```python
answer_json = llm(prompt)
```


```python
print(answer_json)
```

    {
    "action": "ANSWER",
    "answer": "Yes, you can still join the course even after the official start date and submit homeworks. However, be mindful of deadlines for the final projects.",
    "source": "CONTEXT"
    }


## Puttin agentic RAG together


```python
def agentic_rag_v1(question):
    context = "EMPTY"
    prompt = prompt_template.format(question=question, context=context)
    answer_json = llm(prompt)
    answer = json.loads(answer_json)
    print(answer)

    if answer['action'] == 'SEARCH':
        print('need to perform search...')
        search_results = search(question)
        context = build_context(search_results)
        
        prompt = prompt_template.format(question=question, context=context)
        answer_json = llm(prompt)
        answer = json.loads(answer_json)
        print(answer)

    return answer
```


```python
agentic_rag_v1('how do I join the course?')
```

    {'action': 'SEARCH', 'reasoning': 'The user is asking a general question about joining a course. This is a common question that might be covered in an FAQ for any of the courses I support (Machine Learning, MLOps, Data Engineering, LLM Applications).'}
    need to perform search...
    {'action': 'ANSWER', 'answer': 'To join the course, you should register using the provided link before it starts. The course begins on January 15th, 2024, at 17:00 with the first "Office Hours" live session. You can also subscribe to the course\'s public Google Calendar, join the course Telegram channel for announcements, and register in DataTalks.Club\'s Slack to join the channel.', 'source': 'CONTEXT'}





    {'action': 'ANSWER',
     'answer': 'To join the course, you should register using the provided link before it starts. The course begins on January 15th, 2024, at 17:00 with the first "Office Hours" live session. You can also subscribe to the course\'s public Google Calendar, join the course Telegram channel for announcements, and register in DataTalks.Club\'s Slack to join the channel.',
     'source': 'CONTEXT'}




```python
agentic_rag_v1('how patch KDE under FreeBSD?')
```

    {'action': 'SEARCH', 'reasoning': "The question is about patching KDE under FreeBSD, which is an operating system and desktop environment topic. This is not directly related to Machine Learning, MLOps, Data Engineering, or LLM Applications, and therefore likely not in our course FAQ database. A search action is chosen to indicate that it's outside the scope of direct course material but a search would be the next logical step if it were related."}
    need to perform search...
    {'action': 'ANSWER', 'answer': "Patching KDE under FreeBSD typically involves using the FreeBSD Ports or Packages system. The most common and recommended way to update or patch software, including desktop environments like KDE, on FreeBSD is through the `pkg` command for binary packages or by updating and re-installing from the Ports tree. \n\n1.  **Using `pkg` (for binary packages):**\n    If you installed KDE using `pkg install kde5` (or similar), you can update it by running `pkg upgrade`. This command will fetch and install the latest versions of all installed packages, including KDE components, from the official FreeBSD package repositories.\n\n2.  **Using Ports (for source-based installation):**\n    If you built KDE from the Ports tree (i.e., you navigated to `/usr/ports/x11/kde5` and ran `make install` or `make reinstall`), you'll need to update your Ports tree first, and then rebuild/reinstall KDE. \n    *   Update your Ports tree: `portsnap fetch update` (or `git pull` if you're using a Git-based ports tree).\n    *   Navigate to the KDE port directory: `cd /usr/ports/x11/kde5` (or the specific KDE component you want to update, e.g., `x11/plasma5-plasma` for Plasma desktop itself).\n    *   Clean and rebuild/reinstall: `make clean install` or `make clean reinstall`. `make reinstall` is generally safer as it attempts to preserve configuration where possible, but `make install` will also work if the application allows it. You might need to add `BATCH=yes` to the `make` command if you don't want to be prompted for configuration options for all dependencies.\n\n    *Important Note for Ports:* Updating extensively via ports can be time-consuming and sometimes complex due to dependencies. Using `portmaster` or `portupgrade` utilities can significantly simplify the process of managing and updating ports, as they can automate dependency handling and rebuilding.\n\n    *   Install `portmaster`: `pkg install portmaster`\n    *   Update all installed ports: `portmaster -Da` (This will update everything, including KDE).\n\nBefore undertaking any major update, especially with ports, it's always a good idea to back up important configuration files and check the FreeBSD forums, mailing lists, or release notes for any known issues with the specific KDE version you're upgrading to.\n\nIn summary, for most users: `pkg upgrade` is the simplest way to get security patches and updates. For those who compiled from source: update your ports tree and use `portmaster` or `make clean reinstall` on the relevant KDE ports.", 'source': 'OWN_KNOWLEDGE'}





    {'action': 'ANSWER',
     'answer': "Patching KDE under FreeBSD typically involves using the FreeBSD Ports or Packages system. The most common and recommended way to update or patch software, including desktop environments like KDE, on FreeBSD is through the `pkg` command for binary packages or by updating and re-installing from the Ports tree. \n\n1.  **Using `pkg` (for binary packages):**\n    If you installed KDE using `pkg install kde5` (or similar), you can update it by running `pkg upgrade`. This command will fetch and install the latest versions of all installed packages, including KDE components, from the official FreeBSD package repositories.\n\n2.  **Using Ports (for source-based installation):**\n    If you built KDE from the Ports tree (i.e., you navigated to `/usr/ports/x11/kde5` and ran `make install` or `make reinstall`), you'll need to update your Ports tree first, and then rebuild/reinstall KDE. \n    *   Update your Ports tree: `portsnap fetch update` (or `git pull` if you're using a Git-based ports tree).\n    *   Navigate to the KDE port directory: `cd /usr/ports/x11/kde5` (or the specific KDE component you want to update, e.g., `x11/plasma5-plasma` for Plasma desktop itself).\n    *   Clean and rebuild/reinstall: `make clean install` or `make clean reinstall`. `make reinstall` is generally safer as it attempts to preserve configuration where possible, but `make install` will also work if the application allows it. You might need to add `BATCH=yes` to the `make` command if you don't want to be prompted for configuration options for all dependencies.\n\n    *Important Note for Ports:* Updating extensively via ports can be time-consuming and sometimes complex due to dependencies. Using `portmaster` or `portupgrade` utilities can significantly simplify the process of managing and updating ports, as they can automate dependency handling and rebuilding.\n\n    *   Install `portmaster`: `pkg install portmaster`\n    *   Update all installed ports: `portmaster -Da` (This will update everything, including KDE).\n\nBefore undertaking any major update, especially with ports, it's always a good idea to back up important configuration files and check the FreeBSD forums, mailing lists, or release notes for any known issues with the specific KDE version you're upgrading to.\n\nIn summary, for most users: `pkg upgrade` is the simplest way to get security patches and updates. For those who compiled from source: update your ports tree and use `portmaster` or `make clean reinstall` on the relevant KDE ports.",
     'source': 'OWN_KNOWLEDGE'}



# Agentic Search

Agentic Search is an extension of Agentic RAG by leveraging an LLM Agent to formulate search queries based on the initial question.


```python
question = 'how do I do well in module 1'
```


```python
prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information.

Output templates:

If you want to perform search, use this template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer, use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
""".strip()
```


```python
question = "how do I join the course?"

search_queries = []
search_results = []
previous_actions = []
context = build_context(search_results)

prompt = prompt_template.format(
    question=question,
    context=context,
    search_queries="\n".join(search_queries),
    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
    max_iterations=3,
    iteration_number=1
)
print(prompt)
```

    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 1. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    how do I join the course?
    </QUESTION>
    
    <SEARCH_QUERIES>
    
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    
    </PREVIOUS_ACTIONS>



```python
answer_json = llm(prompt)
answer = json.loads(answer_json)
print(json.dumps(answer, indent=2))
```

    {
      "action": "SEARCH",
      "reasoning": "The user is asking how to join the course. I need to search the FAQ for information on course enrollment or joining procedures.",
      "keywords": [
        "join course",
        "enrollment process",
        "how to register for the course"
      ]
    }



```python
previous_actions.append(answer)
```


```python
previous_actions
```




    [{'action': 'SEARCH',
      'reasoning': 'The user is asking how to join the course. I need to search the FAQ for information on course enrollment or joining procedures.',
      'keywords': ['join course',
       'enrollment process',
       'how to register for the course']}]




```python
keywords = answer['keywords']
search_queries.extend(keywords)
```


```python
for k in keywords:
    res = search(k)
    search_results.extend(res)
```


```python
def dedup(seq):
    seen = set()
    result = []
    for el in seq:
        _id = el['_id']
        if _id in seen:
            continue
        seen.add(_id)
        result.append(el)
    return result

search_results = dedup(search_results)
```


```python
context = build_context(search_results)

prompt = prompt_template.format(
    question=question,
    context=context,
    search_queries="\n".join(search_queries),
    previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
    max_iterations=3,
    iteration_number=2
)
print(prompt)
```

    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 2. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    how do I join the course?
    </QUESTION>
    
    <SEARCH_QUERIES>
    join course
    enrollment process
    how to register for the course
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    section: General course-related questions
    question: Course - Can I still join the course after the start date?
    answer: Yes, even if you don't register, you're still eligible to submit the homeworks.
    Be aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.
    
    section: General course-related questions
    question: Course - When will the course start?
    answer: The purpose of this document is to capture frequently asked technical questions
    The exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1
    Subscribe to course public Google Calendar (it works from Desktop only).
    Register before the course starts using this link.
    Join the course Telegram channel with announcements.
    Don’t forget to register in DataTalks.Club's Slack and join the channel.
    
    section: General course-related questions
    question: Course - Can I follow the course after it finishes?
    answer: Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.
    You can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.
    
    section: General course-related questions
    question: Certificate - Can I follow the course in a self-paced mode and get a certificate?
    answer: No, you can only get a certificate if you finish the course with a “live” cohort. We don't award certificates for the self-paced mode. The reason is you need to peer-review capstone(s) after submitting a project. You can only peer-review projects at the time the course is running.
    
    section: General course-related questions
    question: How do I use Git / GitHub for this course?
    answer: After you create a GitHub account, you should clone the course repo to your local machine using the process outlined in this video: Git for Everybody: How to Clone a Repository from GitHub
    Having this local repository on your computer will make it easy for you to access the instructors’ code and make pull requests (if you want to add your own notes or make changes to the course content).
    You will probably also create your own repositories that host your notes, versions of your file, to do this. Here is a great tutorial that shows you how to do this: https://www.atlassian.com/git/tutorials/setting-up-a-repository
    Remember to ignore large database, .csv, and .gz files, and other files that should not be saved to a repository. Use .gitignore for this: https://www.atlassian.com/git/tutorials/saving-changes/gitignore NEVER store passwords or keys in a git repo (even if that repo is set to private).
    This is also a great resource: https://dangitgit.com/
    
    section: Module 5: pyspark
    question: lsRuntimeError: Java gateway process exited before sending its port number
    answer: After installing all including pyspark (and it is successfully imported), but then running this script on the jupyter notebook
    import pyspark
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
    .master("local[*]") \
    .appName('test') \
    .getOrCreate()
    df = spark.read \
    .option("header", "true") \
    .csv('taxi+_zone_lookup.csv')
    df.show()
    it gives the error:
    RuntimeError: Java gateway process exited before sending its port number
    ✅The solution (for me) was:
    pip install findspark on the command line and then
    Add
    import findspark
    findspark.init()
    to the top of the script.
    Another possible solution is:
    Check that pyspark is pointing to the correct location.
    Run pyspark.__file__. It should be list /home/<your user name>/spark/spark-3.0.3-bin-hadoop3.2/python/pyspark/__init__.py if you followed the videos.
    If it is pointing to your python site-packages remove the pyspark directory there and check that you have added the correct exports to you .bashrc file and that there are not any other exports which might supersede the ones provided in the course content.
    To add to the solution above, if the errors persist in regards to setting the correct path for spark,  an alternative solution for permanent path setting solve the error is  to set environment variables on system and user environment variables following this tutorial: Install Apache PySpark on Windows PC | Apache Spark Installation Guide
    Once everything is installed, skip to 7:14 to set up environment variables. This allows for the environment variables to be set permanently.
    
    section: Module 2: Workflow Orchestration
    question: Process to download the VSC using Pandas is killed right away
    answer: pd.read_csv
    df_iter = pd.read_csv(dataset_url, iterator=True, chunksize=100000)
    The data needs to be appended to the parquet file using the fastparquet engine
    df.to_parquet(path, compression="gzip", engine='fastparquet', append=True)
    
    section: Module 1: Docker and Terraform
    question: Docker-Compose - Errors pertaining to docker-compose.yml and pgadmin setup
    answer: For everyone who's having problem with Docker compose, getting the data in postgres and similar issues, please take care of the following:
    create a new volume on docker (either using the command line or docker desktop app)
    make the following changes to your docker-compose.yml file (see attachment)
    set low_memory=false when importing the csv file (df = pd.read_csv('yellow_tripdata_2021-01.csv', nrows=1000, low_memory=False))
    use the below function (in the upload-data.ipynb) for better tracking of your ingestion process (see attachment)
    Order of execution:
    (1) open terminal in 2_docker_sql folder and run docker compose up
    (2) ensure no other containers are running except the one you just executed (pgadmin and pgdatabase)
    (3) open jupyter notebook and begin the data ingestion
    (4) open pgadmin and set up a server (make sure you use the same configurations as your docker-compose.yml file like the same name (pgdatabase), port, databasename (ny_taxi) etc.
    
    section: Module 2: Workflow Orchestration
    question: GCP - 2.2.7d Part 2 - Getting error when you run terraform apply
    answer: If you get the following error
    You have to edit variables.tf on the gcp folder, set your project-id and region and zones properly. Then, run terraform apply again.
    You can find correct regions/zones here: https://cloud.google.com/compute/docs/regions-zones
    Deploying MAGE to GCP  with Terraform via the VM (2.2.7)
    FYI - It can take up to 20 minutes to deploy the MAGE Terraform files if you are using a GCP Virtual Machine. It is normal, so don’t interrupt the process or think it’s taking too long. If you have, make sure you run a terraform destroy before trying again as you will have likely partially created resources which will cause errors next time you run `terraform apply`.
    `terraform destroy` may not completely delete partial resources - go to Google Cloud Console and use the search bar at the top to search for the ‘app.name’ you declared in your variables.tf file; this will list all resources with that name - make sure you delete them all before running `terraform apply` again.
    Why are my GCP free credits going so fast? MAGE .tf files - Terraform Destroy not destroying all Resources
    I checked my GCP billing last night & the MAGE Terraform IaC didn't destroy a GCP Resource called Filestore as ‘mage-data-prep- it has been costing £5.01 of my free credits each day  I now have £151 left - Alexey has assured me that This amount WILL BE SUFFICIENT funds to finish the course. Note to anyone who had issues deploying the MAGE terraform code: check your billing account to see what you're being charged for (main menu - billing) (even if it's your free credits) and run a search for 'mage-data-prep' in the top bar just to be sure that your resources have been destroyed - if any come up delete them.
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    {"action": "SEARCH", "reasoning": "The user is asking how to join the course. I need to search the FAQ for information on course enrollment or joining procedures.", "keywords": ["join course", "enrollment process", "how to register for the course"]}
    </PREVIOUS_ACTIONS>



```python
answer_json = llm(prompt)
answer = json.loads(answer_json)
print(json.dumps(answer, indent=2))
```

    {
      "action": "ANSWER_CONTEXT",
      "answer": "To join the course, you should register using the provided link. You can also subscribe to the course public Google Calendar (from desktop only) and join the course Telegram channel for announcements. Additionally, register in DataTalks.Club's Slack and join the relevant channel. Even if you don't register, you can still submit homeworks, but be aware of deadlines for final projects. If you want a certificate, you must finish the course with a \"live\" cohort, as certificates are not awarded for self-paced mode.",
      "source": "CONTEXT"
    }



```python
question = "what do I need to do to be successful at module 1?"

search_queries = []
search_results = []
previous_actions = []


iteration = 0

while True:
    print(f'ITERATION #{iteration}...')

    context = build_context(search_results)
    prompt = prompt_template.format(
        question=question,
        context=context,
        search_queries="\n".join(search_queries),
        previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
        max_iterations=3,
        iteration_number=iteration
    )

    print(prompt)

    answer_json = llm(prompt)
    answer = json.loads(answer_json)
    print(json.dumps(answer, indent=2))

    previous_actions.append(answer)

    action = answer['action']
    if action != 'SEARCH':
        break

    keywords = answer['keywords']
    search_queries = list(set(search_queries) | set(keywords))
    
    for k in keywords:
        res = search(k)
        search_results.extend(res)

    search_results = dedup(search_results)
    
    iteration = iteration + 1
    if iteration >= 4:
        break

    print()
```

    ITERATION #0...
    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 0. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    what do I need to do to be successful at module 1?
    </QUESTION>
    
    <SEARCH_QUERIES>
    
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    
    </PREVIOUS_ACTIONS>
    {
      "action": "SEARCH",
      "reasoning": "The user is asking for advice on how to be successful in Module 1. I need to search the FAQ database for information specific to Module 1 and general success tips for modules.",
      "keywords": [
        "Module 1 success tips",
        "how to succeed in Module 1",
        "Module 1 requirements",
        "Module 1 study guide"
      ]
    }
    
    ITERATION #1...
    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 1. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    what do I need to do to be successful at module 1?
    </QUESTION>
    
    <SEARCH_QUERIES>
    Module 1 study guide
    Module 1 success tips
    Module 1 requirements
    how to succeed in Module 1
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    section: Module 5: pyspark
    question: Module Not Found Error in Jupyter Notebook .
    answer: Even after installing pyspark correctly on linux machine (VM ) as per course instructions, faced a module not found error in jupyter notebook .
    The solution which worked for me(use following in jupyter notebook) :
    !pip install findspark
    import findspark
    findspark.init()
    Thereafter , import pyspark and create spark contex<<t as usual
    None of the solutions above worked for me till I ran !pip3 install pyspark instead !pip install pyspark.
    Filter based on conditions based on multiple columns
    from pyspark.sql.functions import col
    new_final.filter((new_final.a_zone=="Murray Hill") & (new_final.b_zone=="Midwood")).show()
    Krishna Anand
    
    section: Module 5: pyspark
    question: Py4JJavaError - ModuleNotFoundError: No module named 'py4j'` while executing `import pyspark`
    answer: You need to look for the Py4J file and note the version of the filename. Once you know the version, you can update the export command accordingly, this is how you check yours:
    ` ls ${SPARK_HOME}/python/lib/ ` and then you add it in the export command, mine was:
    export PYTHONPATH=”${SPARK_HOME}/python/lib/Py4J-0.10.9.5-src.zip:${PYTHONPATH}”
    Make sure that the version under `${SPARK_HOME}/python/lib/` matches the filename of py4j or you will encounter `ModuleNotFoundError: No module named 'py4j'` while executing `import pyspark`.
    For instance, if the file under `${SPARK_HOME}/python/lib/` was `py4j-0.10.9.3-src.zip`.
    Then the export PYTHONPATH statement above should be changed to `export PYTHONPATH="${SPARK_HOME}/python/lib/py4j-0.10.9.3-src.zip:$PYTHONPATH"` appropriately.
    Additionally, you can check for the version of ‘py4j’ of the spark you’re using from here and update as mentioned above.
    ~ Abhijit Chakraborty: Sometimes, even with adding the correct version of py4j might not solve the problem. Simply run pip install py4j and problem should be resolved.
    
    section: Module 4: analytics engineering with dbt
    question: DBT - Error: No module named 'pytz' while setting up dbt with docker
    answer: Following dbt with BigQuery on Docker readme.md, after `docker-compose build` and `docker-compose run dbt-bq-dtc init`, encountered error `ModuleNotFoundError: No module named 'pytz'`
    Solution:
    Add `RUN python -m pip install --no-cache pytz` in the Dockerfile under `FROM --platform=$build_for python:3.9.9-slim-bullseye as base`
    
    section: Module 1: Docker and Terraform
    question: Postgres - ModuleNotFoundError: No module named 'psycopg2'
    answer: Issue:
    e…
    Solution:
    pip install psycopg2-binary
    If you already have it, you might need to update it:
    pip install psycopg2-binary --upgrade
    Other methods, if the above fails:
    if you are getting the “ ModuleNotFoundError: No module named 'psycopg2' “ error even after the above installation, then try updating conda using the command conda update -n base -c defaults conda. Or if you are using pip, then try updating it before installing the psycopg packages i.e
    First uninstall the psycopg package
    Then update conda or pip
    Then install psycopg again using pip.
    if you are still facing error with r pcycopg2 and showing pg_config not found then you will have to install postgresql. in MAC it is brew install postgresql
    
    section: Module 1: Docker and Terraform
    question: Python - SQLALchemy - TypeError 'module' object is not callable
    answer: create_engine('postgresql://root:root@localhost:5432/ny_taxi')  I get the error "TypeError: 'module' object is not callable"
    Solution:
    conn_string = "postgresql+psycopg://root:root@localhost:5432/ny_taxi"
    engine = create_engine(conn_string)
    
    section: Module 6: streaming with kafka
    question: Module “kafka” not found when trying to run producer.py
    answer: Solution from Alexey: create a virtual environment and run requirements.txt and the python files in that environment.
    To create a virtual env and install packages (run only once)
    python -m venv env
    source env/bin/activate
    pip install -r ../requirements.txt
    To activate it (you'll need to run it every time you need the virtual env):
    source env/bin/activate
    To deactivate it:
    deactivate
    This works on MacOS, Linux and Windows - but for Windows the path is slightly different (it's env/Scripts/activate)
    Also the virtual environment should be created only to run the python file. Docker images should first all be up and running.
    
    section: Module 1: Docker and Terraform
    question: Python - SQLAlchemy - ModuleNotFoundError: No module named 'psycopg2'.
    answer: Error raised during the jupyter notebook’s cell execution:
    engine = create_engine('postgresql://root:root@localhost:5432/ny_taxi').
    Solution: Need to install Python module “psycopg2”. Can be installed by Conda or pip.
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    {"action": "SEARCH", "reasoning": "The user is asking for advice on how to be successful in Module 1. I need to search the FAQ database for information specific to Module 1 and general success tips for modules.", "keywords": ["Module 1 success tips", "how to succeed in Module 1", "Module 1 requirements", "Module 1 study guide"]}
    </PREVIOUS_ACTIONS>
    {
      "action": "ANSWER_CONTEXT",
      "answer": "To be successful in Module 1, which focuses on Docker and Terraform, you should be aware of and prepared to handle common technical issues. Based on the provided context, some frequently encountered problems and their solutions include:\n\n*   **Postgres - ModuleNotFoundError: No module named 'psycopg2'**: This error often occurs when the `psycopg2` Python module is not installed or needs to be updated. The solution is to run `pip install psycopg2-binary` or `pip install psycopg2-binary --upgrade`. If issues persist, try uninstalling, updating pip/conda, and then reinstalling. For Mac users, installing PostgreSQL via `brew install postgresql` might be necessary if `pg_config` is not found.\n*   **Python - SQLAlchemy - TypeError 'module' object is not callable**: This error, typically encountered when setting up the database engine, can be resolved by correctly formatting the connection string. For example, use `conn_string = \"postgresql+psycopg://root:root@localhost:5432/ny_taxi\"` and then `engine = create_engine(conn_string)`.\n*   **Python - SQLAlchemy - ModuleNotFoundError: No module named 'psycopg2'**: Similar to the Postgres specific `ModuleNotFoundError`, this indicates the `psycopg2` Python module is missing. The solution is to install it using Conda or pip.\n\nWhile the context mainly provides solutions to common technical roadblocks in Module 1, successfully navigating these issues is a key part of progressing through the module. Beyond troubleshooting, general success in any module typically involves understanding the core concepts, actively participating, and practicing the skills taught.",
      "source": "CONTEXT"
    }



```python
def agentic_search(question):
    # Initialize lists to store search queries, results, and previous agent actions
    search_queries = []
    search_results = []
    previous_actions = []

    # Initialize iteration counter
    iteration = 0
    
    # Start the main loop for iterative search
    while True:
        # Print current iteration number for debugging/tracking
        print(f'ITERATION #{iteration}...')
    
        # Build context from previous search results to inform the LLM
        context = build_context(search_results)
        # Format the prompt using a predefined template and current state
        prompt = prompt_template.format(
            question=question,
            context=context,
            search_queries="\n".join(search_queries),
            previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
            max_iterations=3, # Note: This parameter is in the prompt, but the actual loop limit is 4.
            iteration_number=iteration
        )
    
        # Print the prompt sent to the LLM
        print(prompt)
    
        # Call the LLM with the formatted prompt to get an action
        answer_json = llm(prompt)
        # Parse the JSON response from the LLM
        answer = json.loads(answer_json)
        # Print the LLM's response for debugging/tracking
        print(json.dumps(answer, indent=2))

        # Record the current action in previous_actions for future iterations
        previous_actions.append(answer)
    
        # Get the action type from the LLM's response
        action = answer['action']
        # If the action is not 'SEARCH', break the loop (e.g., if it's 'FINISH' or 'ANSWER')
        if action != 'SEARCH':
            break
    
        # Extract keywords for searching from the LLM's response
        keywords = answer['keywords']
        # Update the list of all search queries, ensuring uniqueness
        search_queries = list(set(search_queries) | set(keywords))

        # Perform searches for each new keyword
        for k in keywords:
            res = search(k)
            # Add new results to the overall search_results list
            search_results.extend(res)
    
        # Deduplicate search results to avoid redundant information
        search_results = dedup(search_results)
        
        # Increment the iteration counter
        iteration = iteration + 1
        # Set a hard limit on the number of iterations to prevent infinite loops
        if iteration >= 4:
            break
    
        # Print a newline for better readability between iterations
        print()

    # Return the final answer from the LLM
    return answer
```


```python
answer = agentic_search('how do I prepare for the course?')
```

    ITERATION #0...
    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 0. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    how do I prepare for the course?
    </QUESTION>
    
    <SEARCH_QUERIES>
    
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    
    </PREVIOUS_ACTIONS>
    {
      "action": "SEARCH",
      "reasoning": "The user is asking about course preparation. I should search the FAQ for information related to course preparation, prerequisites, or introductory materials.",
      "keywords": [
        "course preparation",
        "how to prepare for course",
        "prerequisites for course"
      ]
    }
    
    ITERATION #1...
    You're a course teaching assistant.
    
    You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
    
    The CONTEXT is build with the documents from our FAQ database.
    SEARCH_QUERIES contains the queries that were used to retrieve the documents
    from FAQ to and add them to the context.
    PREVIOUS_ACTIONS contains the actions you already performed.
    
    At the beginning the CONTEXT is empty.
    
    You can perform the following actions:
    
    - Search in the FAQ database to get more data for the CONTEXT
    - Answer the question using the CONTEXT
    - Answer the question using your own knowledge
    
    For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
    Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
    
    Don't use search queries used at the previous iterations.
    
    Don't repeat previously performed actions.
    
    Don't perform more than 3 iterations for a given student question.
    The current iteration number: 1. If we exceed the allowed number 
    of iterations, give the best possible answer with the provided information.
    
    Output templates:
    
    If you want to perform search, use this template:
    
    {
    "action": "SEARCH",
    "reasoning": "<add your reasoning here>",
    "keywords": ["search query 1", "search query 2", ...]
    }
    
    If you can answer the QUESTION using CONTEXT, use this template:
    
    {
    "action": "ANSWER_CONTEXT",
    "answer": "<your answer>",
    "source": "CONTEXT"
    }
    
    If the context doesn't contain the answer, use your own knowledge to answer the question
    
    {
    "action": "ANSWER",
    "answer": "<your answer>",
    "source": "OWN_KNOWLEDGE"
    }
    
    <QUESTION>
    how do I prepare for the course?
    </QUESTION>
    
    <SEARCH_QUERIES>
    course preparation
    how to prepare for course
    prerequisites for course
    </SEARCH_QUERIES>
    
    <CONTEXT> 
    section: General course-related questions
    question: Course - When will the course start?
    answer: The purpose of this document is to capture frequently asked technical questions
    The exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1
    Subscribe to course public Google Calendar (it works from Desktop only).
    Register before the course starts using this link.
    Join the course Telegram channel with announcements.
    Don’t forget to register in DataTalks.Club's Slack and join the channel.
    
    section: General course-related questions
    question: Course - Can I follow the course after it finishes?
    answer: Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.
    You can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.
    
    section: General course-related questions
    question: Certificate - Can I follow the course in a self-paced mode and get a certificate?
    answer: No, you can only get a certificate if you finish the course with a “live” cohort. We don't award certificates for the self-paced mode. The reason is you need to peer-review capstone(s) after submitting a project. You can only peer-review projects at the time the course is running.
    
    section: General course-related questions
    question: Is it possible to use tool “X” instead of the one tool you use in the course?
    answer: Yes, this applies if you want to use Airflow or Prefect instead of Mage, AWS or Snowflake instead of GCP products or Tableau instead of Metabase or Google data studio.
    The course covers 2 alternative data stacks, one using GCP and one using local installation of everything. You can use one of them or use your tool of choice.
    Should you consider it instead of the one tool you use? That we can’t support you if you choose to use a different stack, also you would need to explain the different choices of tool for the peer review of your capstone project.
    
    section: General course-related questions
    question: How do I use Git / GitHub for this course?
    answer: After you create a GitHub account, you should clone the course repo to your local machine using the process outlined in this video: Git for Everybody: How to Clone a Repository from GitHub
    Having this local repository on your computer will make it easy for you to access the instructors’ code and make pull requests (if you want to add your own notes or make changes to the course content).
    You will probably also create your own repositories that host your notes, versions of your file, to do this. Here is a great tutorial that shows you how to do this: https://www.atlassian.com/git/tutorials/setting-up-a-repository
    Remember to ignore large database, .csv, and .gz files, and other files that should not be saved to a repository. Use .gitignore for this: https://www.atlassian.com/git/tutorials/saving-changes/gitignore NEVER store passwords or keys in a git repo (even if that repo is set to private).
    This is also a great resource: https://dangitgit.com/
    
    section: General course-related questions
    question: Course - What are the prerequisites for this course?
    answer: GitHub - DataTalksClub data-engineering-zoomcamp#prerequisites
    
    section: General course-related questions
    question: Course - What can I do before the course starts?
    answer: You can start by installing and setting up all the dependencies and requirements:
    Google cloud account
    Google Cloud SDK
    Python 3 (installed with Anaconda)
    Terraform
    Git
    Look over the prerequisites and syllabus to see if you are comfortable with these subjects.
    </CONTEXT>
    
    <PREVIOUS_ACTIONS>
    {"action": "SEARCH", "reasoning": "The user is asking about course preparation. I should search the FAQ for information related to course preparation, prerequisites, or introductory materials.", "keywords": ["course preparation", "how to prepare for course", "prerequisites for course"]}
    </PREVIOUS_ACTIONS>
    {
      "action": "ANSWER_CONTEXT",
      "answer": "To prepare for the course, you can start by installing and setting up the necessary dependencies and requirements, which include a Google Cloud account, Google Cloud SDK, Python 3 (installed with Anaconda), Terraform, and Git. It's also recommended to review the prerequisites and syllabus to ensure you are comfortable with the subjects covered.\n\nAdditionally, you should clone the course repository from GitHub to your local machine. For using Git and GitHub, resources like \"Git for Everybody: How to Clone a Repository from GitHub\" and a tutorial on setting up a repository (https://www.atlassian.com/git/tutorials/setting-up-a-repository) are suggested. Remember to use `.gitignore` for large files and never store passwords or keys in a Git repository.",
      "source": "CONTEXT"
    }



```python
from IPython.display import display_markdown, Markdown
```


```python
Markdown(answer['answer'])
```




To prepare for the course, you can start by installing and setting up the necessary dependencies and requirements, which include a Google Cloud account, Google Cloud SDK, Python 3 (installed with Anaconda), Terraform, and Git. It's also recommended to review the prerequisites and syllabus to ensure you are comfortable with the subjects covered.

Additionally, you should clone the course repository from GitHub to your local machine. For using Git and GitHub, resources like "Git for Everybody: How to Clone a Repository from GitHub" and a tutorial on setting up a repository (https://www.atlassian.com/git/tutorials/setting-up-a-repository) are suggested. Remember to use `.gitignore` for large files and never store passwords or keys in a Git repository.



[Brave Search API](https://brave.com/search/api/)

# Function calling

```python
def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results
```

Can be jsonified for OpenAI like :

```json
search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

Slightly different syntax for OpenRouter (older OpenAI syntax ?):

http://openrouter.ai/docs/features/tool-calling

```json
search_tool = {
    "type": "function",
    "function":{
        "name": "search",
        "description": "Search the FAQ database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}
```


```python
search_tool = {
    "type": "function",
    "function":{
        "name": "search",
        "description": "Search the FAQ database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}
```


```python
question = "How to do well in Module 1 ?"

developer_prompt = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.
""".strip()

tools = [search_tool]

chat_messages = [
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": question}
]

response = client.chat.completions.create(
    model=MODEL,
    messages=chat_messages,
    tools=tools
)

```


```python
call = response.choices[0].message.tool_calls[0]
call
```




    ChatCompletionMessageToolCall(id='tool_0_search', function=Function(arguments='{"query":"how to do well in Module 1"}', name='search'), type='function', index=0)




```python
call_id = call.id
call_id
```




    'tool_0_search'




```python
f_name = call.function.name
arguments  = json.loads(
    call.function.arguments
)
print(f_name)
print(arguments)
```

    search
    {'query': 'how to do well in Module 1'}



```python
# get functions by name from globals()
f = globals()[f_name]
```


```python
results = f(**arguments)
```


```python
search_results = json.dumps(results, indent=2)
print(search_results[:100])
```

    [
      {
        "text": "Even after installing pyspark correctly on linux machine (VM ) as per course inst



```python
chat_messages.append(call)

chat_messages.append({
    "role": "tool",
    "tool_call_id": call_id,
    "name":f_name,
    "content": json.dumps(
        search_results
    )
})
```


```python
chat_messages
```




    [{'role': 'developer',
      'content': "You're a course teaching assistant.\nYou're given a question from a course student and your task is to answer it."},
     {'role': 'user', 'content': 'How to do well in Module 1 ?'},
     ChatCompletionMessageToolCall(id='tool_0_search', function=Function(arguments='{"query":"How to do well in Module 1"}', name='search'), type='function', index=0),
     {'role': 'tool',
      'tool_call_id': 'tool_0_search',
      'name': 'search',
      'content': '"[\\n  {\\n    \\"text\\": \\"Even after installing pyspark correctly on linux machine (VM ) as per course instructions, faced a module not found error in jupyter notebook .\\\\nThe solution which worked for me(use following in jupyter notebook) :\\\\n!pip install findspark\\\\nimport findspark\\\\nfindspark.init()\\\\nThereafter , import pyspark and create spark contex<<t as usual\\\\nNone of the solutions above worked for me till I ran !pip3 install pyspark instead !pip install pyspark.\\\\nFilter based on conditions based on multiple columns\\\\nfrom pyspark.sql.functions import col\\\\nnew_final.filter((new_final.a_zone==\\\\\\"Murray Hill\\\\\\") & (new_final.b_zone==\\\\\\"Midwood\\\\\\")).show()\\\\nKrishna Anand\\",\\n    \\"section\\": \\"Module 5: pyspark\\",\\n    \\"question\\": \\"Module Not Found Error in Jupyter Notebook .\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 322\\n  },\\n  {\\n    \\"text\\": \\"You need to look for the Py4J file and note the version of the filename. Once you know the version, you can update the export command accordingly, this is how you check yours:\\\\n` ls ${SPARK_HOME}/python/lib/ ` and then you add it in the export command, mine was:\\\\nexport PYTHONPATH=\\\\u201d${SPARK_HOME}/python/lib/Py4J-0.10.9.5-src.zip:${PYTHONPATH}\\\\u201d\\\\nMake sure that the version under `${SPARK_HOME}/python/lib/` matches the filename of py4j or you will encounter `ModuleNotFoundError: No module named \'py4j\'` while executing `import pyspark`.\\\\nFor instance, if the file under `${SPARK_HOME}/python/lib/` was `py4j-0.10.9.3-src.zip`.\\\\nThen the export PYTHONPATH statement above should be changed to `export PYTHONPATH=\\\\\\"${SPARK_HOME}/python/lib/py4j-0.10.9.3-src.zip:$PYTHONPATH\\\\\\"` appropriately.\\\\nAdditionally, you can check for the version of \\\\u2018py4j\\\\u2019 of the spark you\\\\u2019re using from here and update as mentioned above.\\\\n~ Abhijit Chakraborty: Sometimes, even with adding the correct version of py4j might not solve the problem. Simply run pip install py4j and problem should be resolved.\\",\\n    \\"section\\": \\"Module 5: pyspark\\",\\n    \\"question\\": \\"Py4JJavaError - ModuleNotFoundError: No module named \'py4j\'` while executing `import pyspark`\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 323\\n  },\\n  {\\n    \\"text\\": \\"Following dbt with BigQuery on Docker readme.md, after `docker-compose build` and `docker-compose run dbt-bq-dtc init`, encountered error `ModuleNotFoundError: No module named \'pytz\'`\\\\nSolution:\\\\nAdd `RUN python -m pip install --no-cache pytz` in the Dockerfile under `FROM --platform=$build_for python:3.9.9-slim-bullseye as base`\\",\\n    \\"section\\": \\"Module 4: analytics engineering with dbt\\",\\n    \\"question\\": \\"DBT - Error: No module named \'pytz\' while setting up dbt with docker\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 299\\n  },\\n  {\\n    \\"text\\": \\"create_engine(\'postgresql://root:root@localhost:5432/ny_taxi\')  I get the error \\\\\\"TypeError: \'module\' object is not callable\\\\\\"\\\\nSolution:\\\\nconn_string = \\\\\\"postgresql+psycopg://root:root@localhost:5432/ny_taxi\\\\\\"\\\\nengine = create_engine(conn_string)\\",\\n    \\"section\\": \\"Module 1: Docker and Terraform\\",\\n    \\"question\\": \\"Python - SQLALchemy - TypeError \'module\' object is not callable\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 124\\n  },\\n  {\\n    \\"text\\": \\"Error raised during the jupyter notebook\\\\u2019s cell execution:\\\\nengine = create_engine(\'postgresql://root:root@localhost:5432/ny_taxi\').\\\\nSolution: Need to install Python module \\\\u201cpsycopg2\\\\u201d. Can be installed by Conda or pip.\\",\\n    \\"section\\": \\"Module 1: Docker and Terraform\\",\\n    \\"question\\": \\"Python - SQLAlchemy - ModuleNotFoundError: No module named \'psycopg2\'.\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 125\\n  }\\n]"'},
     ChatCompletionMessageToolCall(id='tool_0_search', function=Function(arguments='{"query":"how to do well in Module 1"}', name='search'), type='function', index=0),
     {'role': 'tool',
      'tool_call_id': 'tool_0_search',
      'name': 'search',
      'content': '"[\\n  {\\n    \\"text\\": \\"Even after installing pyspark correctly on linux machine (VM ) as per course instructions, faced a module not found error in jupyter notebook .\\\\nThe solution which worked for me(use following in jupyter notebook) :\\\\n!pip install findspark\\\\nimport findspark\\\\nfindspark.init()\\\\nThereafter , import pyspark and create spark contex<<t as usual\\\\nNone of the solutions above worked for me till I ran !pip3 install pyspark instead !pip install pyspark.\\\\nFilter based on conditions based on multiple columns\\\\nfrom pyspark.sql.functions import col\\\\nnew_final.filter((new_final.a_zone==\\\\\\"Murray Hill\\\\\\") & (new_final.b_zone==\\\\\\"Midwood\\\\\\")).show()\\\\nKrishna Anand\\",\\n    \\"section\\": \\"Module 5: pyspark\\",\\n    \\"question\\": \\"Module Not Found Error in Jupyter Notebook .\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 322\\n  },\\n  {\\n    \\"text\\": \\"You need to look for the Py4J file and note the version of the filename. Once you know the version, you can update the export command accordingly, this is how you check yours:\\\\n` ls ${SPARK_HOME}/python/lib/ ` and then you add it in the export command, mine was:\\\\nexport PYTHONPATH=\\\\u201d${SPARK_HOME}/python/lib/Py4J-0.10.9.5-src.zip:${PYTHONPATH}\\\\u201d\\\\nMake sure that the version under `${SPARK_HOME}/python/lib/` matches the filename of py4j or you will encounter `ModuleNotFoundError: No module named \'py4j\'` while executing `import pyspark`.\\\\nFor instance, if the file under `${SPARK_HOME}/python/lib/` was `py4j-0.10.9.3-src.zip`.\\\\nThen the export PYTHONPATH statement above should be changed to `export PYTHONPATH=\\\\\\"${SPARK_HOME}/python/lib/py4j-0.10.9.3-src.zip:$PYTHONPATH\\\\\\"` appropriately.\\\\nAdditionally, you can check for the version of \\\\u2018py4j\\\\u2019 of the spark you\\\\u2019re using from here and update as mentioned above.\\\\n~ Abhijit Chakraborty: Sometimes, even with adding the correct version of py4j might not solve the problem. Simply run pip install py4j and problem should be resolved.\\",\\n    \\"section\\": \\"Module 5: pyspark\\",\\n    \\"question\\": \\"Py4JJavaError - ModuleNotFoundError: No module named \'py4j\'` while executing `import pyspark`\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 323\\n  },\\n  {\\n    \\"text\\": \\"Following dbt with BigQuery on Docker readme.md, after `docker-compose build` and `docker-compose run dbt-bq-dtc init`, encountered error `ModuleNotFoundError: No module named \'pytz\'`\\\\nSolution:\\\\nAdd `RUN python -m pip install --no-cache pytz` in the Dockerfile under `FROM --platform=$build_for python:3.9.9-slim-bullseye as base`\\",\\n    \\"section\\": \\"Module 4: analytics engineering with dbt\\",\\n    \\"question\\": \\"DBT - Error: No module named \'pytz\' while setting up dbt with docker\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 299\\n  },\\n  {\\n    \\"text\\": \\"create_engine(\'postgresql://root:root@localhost:5432/ny_taxi\')  I get the error \\\\\\"TypeError: \'module\' object is not callable\\\\\\"\\\\nSolution:\\\\nconn_string = \\\\\\"postgresql+psycopg://root:root@localhost:5432/ny_taxi\\\\\\"\\\\nengine = create_engine(conn_string)\\",\\n    \\"section\\": \\"Module 1: Docker and Terraform\\",\\n    \\"question\\": \\"Python - SQLALchemy - TypeError \'module\' object is not callable\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 124\\n  },\\n  {\\n    \\"text\\": \\"Error raised during the jupyter notebook\\\\u2019s cell execution:\\\\nengine = create_engine(\'postgresql://root:root@localhost:5432/ny_taxi\').\\\\nSolution: Need to install Python module \\\\u201cpsycopg2\\\\u201d. Can be installed by Conda or pip.\\",\\n    \\"section\\": \\"Module 1: Docker and Terraform\\",\\n    \\"question\\": \\"Python - SQLAlchemy - ModuleNotFoundError: No module named \'psycopg2\'.\\",\\n    \\"course\\": \\"data-engineering-zoomcamp\\",\\n    \\"_id\\": 125\\n  }\\n]"'}]




```python
response = client.chat.completions.create(
    model=MODEL,
    messages=chat_messages,
    tools=tools
)
```


```python
Markdown(
    response.choices[0].message.content
)
```




I can't provide specific advice on how to "do well" in Module 1 as the search results primarily focus on troubleshooting technical errors related to different modules (Module 5: PySpark, Module 4: dbt, and Module 1: Docker and Terraform). 

However, looking at the Module 1 entries, these are common issues:

*   **`TypeError: 'module' object is not callable` with SQLAlchemy:** This usually means you're trying to call a module as if it were a function. The solution provided suggests changing the connection string to `postgresql+psycopg://` and then correctly creating the engine.
*   **`ModuleNotFoundError: No module named 'psycopg2'`**: This error indicates that the `psycopg2` Python module, necessary for interacting with PostgreSQL, is not installed. The solution is to install it using `conda` or `pip`.

To do well in Module 1, I would recommend:

1.  **Carefully review the module materials:** Understand the core concepts of Docker and Terraform.
2.  **Pay close attention to setup instructions:** Many errors stem from incorrect environments or missing dependencies.
3.  **Troubleshoot methodically:** When you encounter an error, try to understand the error message.
4.  **Look for similar issues:** If you encounter a problem, it's likely others have too. Search the course FAQs, forums, or general programming resources.

If you have a more specific question about a concept or a problem you're facing in Module 1, feel free to ask!



## Making multiple calls


```python
question = "How to do well in Module 1 ?"

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.
If you look up something in FAQ, convert the student question into multiple queries.
""".strip()

tools = [search_tool]

chat_messages = [
    {"role": "developer", "content": developer_prompt},
    {"role": "user", "content": question}
]
# response will contain multiple tool calls
response = client.chat.completions.create(
    model=MODEL,
    messages=chat_messages,
    tools=tools
)
```


```python
def do_call(tool_call_response):
    call_id = tool_call_response.id
    function_name = tool_call_response.function.name
    arguments = json.loads(tool_call_response.function.arguments)

    f = globals()[function_name]
    result = f(**arguments)

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name":function_name,
        "content": json.dumps(
            result, indent = 2
            )
    }
```


```python
calls = response.choices[0].message.tool_calls
```


```python
for call in calls:
    chat_messages.append(call)
    
    result = do_call(call)
    chat_messages.append(result)
```


```python
# response will contain multiple tool calls
response = client.chat.completions.create(
    model=MODEL,
    messages=chat_messages,
    tools=tools
)
```


```python
Markdown(
    response.choices[0].message.content
)
```




I can't provide specific advice on how to "do well" in Module 1, as that depends on the course content and your learning style. However, I can help you with specific technical issues you might encounter in Module 1.

Are you running into any particular errors or having trouble with any of the concepts? For example, some common issues in Module 1 of the Data Engineering Zoomcamp course include:

*   **`ModuleNotFoundError: No module named 'psycopg2'`**: This error occurs when the `psycopg2` Python module, needed for PostgreSQL, is not installed. You can usually fix this by running `pip install psycopg2-binary`.
*   **`TypeError: 'module' object is not callable` with SQLAlchemy**: This error often means you're trying to call `create_engine` incorrectly. Make sure you are using `create_engine` with the correct connection string format, e.g., `postgresql+psycopg://user:password@host:port/database_name`.

If you have other questions related to specific errors or topics in Module 1, please let me know!




```python
print(response.choices[0].finish_reason)
```

    tool_calls



```python
developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.
When using FAQ, perform deep topic exploration: make one request to FAQ,
and then based on the results, make more requests.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
]
```


```python
while True: # main Q&A loop
    question = input() # How do I do my best for module 1?
    if question == 'stop':
        break

    message = {"role": "user", "content": question}
    chat_messages.append(message)

    while True: # request-response loop - query API till get a message
        # response will contain multiple tool calls
        response = client.chat.completions.create(
            model=MODEL,
            messages=chat_messages,
            tools=tools
        )

        # check for tool calls
        has_tool_calls = response.choices[0].finish_reason == 'tool_calls'
        
        if has_tool_calls:
            
            # consume all tool calls
            calls = response.choices[0].message.tool_calls
            
            for call in calls:
                chat_messages.append(call)
                
                result = do_call(call)
                chat_messages.append(result)
            
        if response.choices[0].finish_reason == 'stop':
            display(
                Markdown(
                    response.choices[0].message.content
                )
            )
            break
        
```


Module 1 of the Data Engineering Zoomcamp covers Docker and Terraform. This includes topics like ingesting NY Taxi Data to Postgres, and resolving `psycopg2` and SQLAlchemy-related errors.

Module 4 focuses on Analytics Engineering with dbt. This module addresses issues like connecting dbt Cloud with BigQuery, resolving `NoneType` object errors in dbt macros, and handling `ModuleNotFoundError: No module named 'pytz'` when setting up dbt with Docker. It also covers what to do if `dbt run` doesn't update your BigQuery table as expected.

What else would you like to know about these modules or any other part of the course?



Module 4 of the Data Engineering Zoomcamp covers "Analytics Engineering with dbt". Here are some key takeaways:

*   **dbt Project Setup:** Ensure your `dbt_project.yml` file is correctly configured, especially if your dbt project is in a subdirectory.
*   **Connecting dbt Cloud with BigQuery:** You might encounter "Access Denied" errors. To resolve this, grant your dbt service account the "BigQuery Job User" role in Google Cloud's IAM & Admin. It's also recommended to add "BigQuery Data Owner," "Storage Object Admin," and "Storage Admin" roles to prevent future permission issues.
*   **Handling Data Types:** Be mindful of column data types when working with dbt and BigQuery, especially when converting from CSV to Parquet. If dbt infers incorrect types, define the schema explicitly in your data ingestion pipeline or cast the columns within your dbt models. Pandas `Int64` (with a capital 'I') can be used for nullable integers when reading CSVs.
*   **Variable Usage:** When using variables with `dbt run`, ensure proper YAML dictionary format, e.g., `dbt run --var 'is_test_run: false'`.
*   **CI/CD Jobs in dbt Cloud:** If "Triggered by pull requests" is disabled when setting up a Continuous Integration job, it's likely due to being on the Developer Plan. You'll need to upgrade to a Team Plan or Enterprise Plan to utilize CI Jobs.

Do you have any specific questions about any of these dbt topics, or would you like to explore another module?



`dbt run` compiles and executes all models in your dbt project, creating or updating the corresponding tables or views in your data warehouse. `dbt build` is a superset of `dbt run`; it not only runs your models but also includes `dbt test`, `dbt snapshot`, and `dbt seed` operations. This means `dbt build` will execute your tests, apply any data snapshots, and load any seed files in addition to building your models.

Here's why you might choose one over the other:

*   **`dbt run`**: Ideal for everyday development when you're primarily focused on building and transforming your data models. It's faster as it skips tests and other operations.
*   **`dbt build`**: Recommended for a more comprehensive workflow, especially before deploying to production or when you want to ensure data quality and integrity. It provides a more robust check of your dbt project.

Do you have any specific issues you're encountering with `dbt run` or `dbt build`?



```python
def add_entry(question, answer):
    doc = {
        'question': question,
        'text': answer,
        'section': 'user added',
        'course': 'data-engineering-zoomcamp'
    }
    index.append(doc)
```


```python
add_entry_description = {
    "type": "function",
    "function": {"name": "add_entry",
    "description": "Add an entry to the FAQ database",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to be added to the FAQ database",
            },
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            }
        },
        "required": ["question", "answer"],
        "additionalProperties": False
    }}
}
```


```python
import chat_assistant
from importlib import reload
chat_assistant = reload(chat_assistant)

tools = chat_assistant.Tools()
tools.add_tool(search, search_tool)
tools.add_tool(add_entry, add_entry_description)
print(tools.get_tools())

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ to augment your answer.

At the end of each response, ask the user a follow up question based on your answer.

Use the add_entry tool to add entries to the FAQ database when asked to do so.
""".strip()

chat_interface = chat_assistant.ChatInterface()

chat = chat_assistant.ChatAssistant(
    tools=tools,
    developer_prompt=developer_prompt,
    chat_interface=chat_interface,
    client=client,
    model=MODEL
)
```

    [{'type': 'function', 'function': {'name': 'search', 'description': 'Search the FAQ database', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': 'Search query text to look up in the course FAQ.'}}, 'required': ['query'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'add_entry', 'description': 'Add an entry to the FAQ database', 'parameters': {'type': 'object', 'properties': {'question': {'type': 'string', 'description': 'The question to be added to the FAQ database'}, 'answer': {'type': 'string', 'description': 'The answer to the question'}}, 'required': ['question', 'answer'], 'additionalProperties': False}}}]



```python
chat.run()
```



            <details>
            <summary>Function call: <tt>search({"query":"module 1"})</tt></summary>
            <div>
                <b>Call</b>
                <pre>ChatCompletionMessageToolCall(id='tool_0_search', function=Function(arguments='{"query":"module 1"}', name='search'), type='function', index=0)</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>[
  {
    "text": "You need to look for the Py4J file and note the version of the filename. Once you know the version, you can update the export command accordingly, this is how you check yours:\n` ls ${SPARK_HOME}/python/lib/ ` and then you add it in the export command, mine was:\nexport PYTHONPATH=\u201d${SPARK_HOME}/python/lib/Py4J-0.10.9.5-src.zip:${PYTHONPATH}\u201d\nMake sure that the version under `${SPARK_HOME}/python/lib/` matches the filename of py4j or you will encounter `ModuleNotFoundError: No module named 'py4j'` while executing `import pyspark`.\nFor instance, if the file under `${SPARK_HOME}/python/lib/` was `py4j-0.10.9.3-src.zip`.\nThen the export PYTHONPATH statement above should be changed to `export PYTHONPATH=\"${SPARK_HOME}/python/lib/py4j-0.10.9.3-src.zip:$PYTHONPATH\"` appropriately.\nAdditionally, you can check for the version of \u2018py4j\u2019 of the spark you\u2019re using from here and update as mentioned above.\n~ Abhijit Chakraborty: Sometimes, even with adding the correct version of py4j might not solve the problem. Simply run pip install py4j and problem should be resolved.",
    "section": "Module 5: pyspark",
    "question": "Py4JJavaError - ModuleNotFoundError: No module named 'py4j'` while executing `import pyspark`",
    "course": "data-engineering-zoomcamp",
    "_id": 323
  },
  {
    "text": "Following dbt with BigQuery on Docker readme.md, after `docker-compose build` and `docker-compose run dbt-bq-dtc init`, encountered error `ModuleNotFoundError: No module named 'pytz'`\nSolution:\nAdd `RUN python -m pip install --no-cache pytz` in the Dockerfile under `FROM --platform=$build_for python:3.9.9-slim-bullseye as base`",
    "section": "Module 4: analytics engineering with dbt",
    "question": "DBT - Error: No module named 'pytz' while setting up dbt with docker",
    "course": "data-engineering-zoomcamp",
    "_id": 299
  },
  {
    "text": "Even after installing pyspark correctly on linux machine (VM ) as per course instructions, faced a module not found error in jupyter notebook .\nThe solution which worked for me(use following in jupyter notebook) :\n!pip install findspark\nimport findspark\nfindspark.init()\nThereafter , import pyspark and create spark contex<<t as usual\nNone of the solutions above worked for me till I ran !pip3 install pyspark instead !pip install pyspark.\nFilter based on conditions based on multiple columns\nfrom pyspark.sql.functions import col\nnew_final.filter((new_final.a_zone==\"Murray Hill\") & (new_final.b_zone==\"Midwood\")).show()\nKrishna Anand",
    "section": "Module 5: pyspark",
    "question": "Module Not Found Error in Jupyter Notebook .",
    "course": "data-engineering-zoomcamp",
    "_id": 322
  },
  {
    "text": "Issue:\ne\u2026\nSolution:\npip install psycopg2-binary\nIf you already have it, you might need to update it:\npip install psycopg2-binary --upgrade\nOther methods, if the above fails:\nif you are getting the \u201c ModuleNotFoundError: No module named 'psycopg2' \u201c error even after the above installation, then try updating conda using the command conda update -n base -c defaults conda. Or if you are using pip, then try updating it before installing the psycopg packages i.e\nFirst uninstall the psycopg package\nThen update conda or pip\nThen install psycopg again using pip.\nif you are still facing error with r pcycopg2 and showing pg_config not found then you will have to install postgresql. in MAC it is brew install postgresql",
    "section": "Module 1: Docker and Terraform",
    "question": "Postgres - ModuleNotFoundError: No module named 'psycopg2'",
    "course": "data-engineering-zoomcamp",
    "_id": 112
  },
  {
    "text": "create_engine('postgresql://root:root@localhost:5432/ny_taxi')  I get the error \"TypeError: 'module' object is not callable\"\nSolution:\nconn_string = \"postgresql+psycopg://root:root@localhost:5432/ny_taxi\"\nengine = create_engine(conn_string)",
    "section": "Module 1: Docker and Terraform",
    "question": "Python - SQLALchemy - TypeError 'module' object is not callable",
    "course": "data-engineering-zoomcamp",
    "_id": 124
  }
]</pre>
            </div>

            </details>





            <div>
                <div><b>Assistant:</b></div>
                <div><p>Module 1 of the course covers Docker and Terraform. It includes setting up your environment and using these tools for data engineering tasks. Some common issues students face in this module include <code>ModuleNotFoundError: No module named 'psycopg2'</code> and <code>TypeError: 'module' object is not callable</code> when working with SQLAlchemy and PostgreSQL.</p>
<p>To resolve the <code>psycopg2</code> error, you can try <code>pip install psycopg2-binary</code> or upgrade your existing installation. If you're still facing issues, ensure your <code>conda</code> or <code>pip</code> is updated, then uninstall and reinstall the package. For the SQLAlchemy <code>TypeError</code>, ensure you're using <code>create_engine</code> correctly with the connection string, for example: <code>engine = create_engine("postgresql+psycopg://root:root@localhost:5432/ny_taxi")</code>.</p>
<p>What specific topics in Docker or Terraform are you interested in learning more about?</p></div>
            </div>





<details>
<summary>Function call: <tt>add_entry({"question":"What is covered in Module 1?","ans...)</tt></summary>
<div>
    <b>Call</b>
    <pre>ChatCompletionMessageToolCall(id='tool_0_add_entry', function=Function(arguments='{"question":"What is covered in Module 1?","answer":"Module 1 of the course covers Docker and Terraform. It includes setting up your environment and using these tools for data engineering tasks. Common issues and solutions for this module are:\\n- `ModuleNotFoundError: No module named \'psycopg2\'`: Try `pip install psycopg2-binary` or upgrade your existing installation. If issues persist, update `conda` or `pip`, then uninstall and reinstall the package.\\n- `TypeError: \'module\' object is not callable` with SQLAlchemy: Ensure correct `create_engine` usage, e.g., `engine = create_engine(\\"postgresql+psycopg://root:root@localhost:5432/ny_taxi\\")`."}', name='add_entry'), type='function', index=0)</pre>
</div>
<div>
    <b>Output</b>
    <pre>null</pre>
</div>

</details>





            <div>
                <div><b>Assistant:</b></div>
                <div><p>I have added the information about Module 1, Docker, and Terraform, along with common issues and their solutions, to the FAQ database.</p>
<p>What other modules or topics are you curious about?</p></div>
            </div>



    Chat ended.


https://ai.pydantic.dev/models/openai/#openrouter


```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.models.openai import OpenAIModel
```


```python
model = OpenAIModel(
    MODEL,  
    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY),
)
```


```python
chat_agent = Agent(
    model,
    system_prompt=developer_prompt
)
```


```python
from typing import Dict


@chat_agent.tool
def search_tool(ctx: RunContext, query: str) -> Dict[str, str]:
    """
    Search the FAQ for relevant entries matching the query.

    Parameters
    ----------
    query : str
        The search query string provided by the user.

    Returns
    -------
    list
        A list of search results (up to 5), each containing relevance information 
        and associated output IDs.
    """
    print(f"search('{query}')")
    return search(query)


@chat_agent.tool
def add_entry_tool(ctx: RunContext, question: str, answer: str) -> None:
    """
    Add a new question-answer entry to FAQ.

    This function creates a document with the given question and answer, 
    tagging it as user-added content.

    Parameters
    ----------
    question : str
        The question text to be added to the index.

    answer : str
        The answer or explanation corresponding to the question.

    Returns
    -------
    None
    """
    return add_entry(question, answer)
```


```python
user_prompt = "what are the tools used in the course and what are common issues with these tools ? use as many search queries as needed"
agent_run = await chat_agent.run(user_prompt)
Markdown(agent_run.output)
```

    search('tools used in the course')
    search('common issues with tools')





The course covers two alternative data stacks: one using Google Cloud Platform (GCP) and another using local installations of various tools. You have the flexibility to choose one of these or even use your own preferred tools, although support will be limited if you deviate from the course's recommended stacks.

Here are some of the tools mentioned:

*   **Orchestration:** Airflow, Prefect, or Mage
*   **Cloud Providers:** AWS or GCP products
*   **Business Intelligence:** Tableau, Metabase, or Google Data Studio
*   **Version Control:** Git/GitHub
*   **Containerization:** Docker
*   **Infrastructure as Code:** Terraform
*   **Database related:** Pandas, SQLAlchemy, psycopg2

Common issues that students encounter with these tools, and general troubleshooting tips, include:

*   **`ModuleNotFoundError: No module named 'psycopg2'`**: This often happens when `psycopg2` is not installed correctly. Try `pip install psycopg2-binary` or upgrade your existing installation. If problems persist, update `conda` or `pip`, then try uninstalling and reinstalling the package.
*   **`TypeError: 'module' object is not callable` with SQLAlchemy**: Ensure you are using `create_engine` correctly, for example: `engine = create_engine("postgresql+psycopg://root:root@localhost:5432/ny_taxi")`.
*   **Pandas missing value issues**: When injecting data into tools like BigQuery, you might face type errors because pandas can parse integer columns with missing values as float types by default. To address this, you can specify or cast the data type to `Int64` during data transformation. Alternatively, you can use `df.convert_dtypes()` after filling missing values (e.g., `df.fillna(-999999, inplace=True)`).
*   **Google Cloud SDK PATH issues on Windows**: If you're consistently getting errors about the installer being unable to automatically update your system PATH, you might need to add Git Bash to your Windows path. This can be done by installing Anaconda Navigator and ensuring you check "add conda to the path" during installation, and/or installing Git Bash and checking "Add GitBash to Windows Terminal" and "Use Git and optional Unix tools from the command prompt." Then, from within Git Bash, run `conda init bash`.
*   **Troubleshooting in general**:
    *   Start by trying to solve the issue yourself by reading error messages and documentation.
    *   Restart the application, server, or PC.
    *   Search for the error online using specific keywords (e.g., `<technology> <problem statement>`).
    *   Check the tool's official documentation.
    *   Consider uninstalling and reinstalling the application.
    *   If all else fails, ask for help in the Slack channel, providing as much detail as possible (OS, commands run, exact error messages, what you've already tried). **Always copy and paste error messages, do not use screenshots.**

What specific tools are you most interested in learning more about, or what kind of issues have you encountered so far?




```python

```
