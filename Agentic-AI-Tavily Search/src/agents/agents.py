from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic


load_dotenv()

# Model Initialization
#llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

# from langchain import HuggingFaceHub
# HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")
# llm2=HuggingFaceHub(repo_id="google/flan-t5-large",huggingfacehub_api_token=HUGGINGFACE_TOKEN)
# llm3=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",huggingfacehub_api_token=HUGGINGFACE_TOKEN)

#https://groq.com/  -- Developers -- Free API Key
# llm = ChatGroq(
#     model="qwen/qwen3-32b",
#     temperature=0   
# )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0   
)



# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
       
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url],

    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

