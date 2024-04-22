
import os
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

def client():

    openai_client = AzureChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY,
            # openai_api_base=OPENAI_API_BASE,
            openai_api_type='azure',
            openai_api_version='2023-09-01-preview',
            deployment_name='MetaEdge_LLM'
        )

    return openai_client

def llm_chain():

    # information = "\n\n".join(retrieved_documents)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content = "你是一名公司的員工，正在針對員工工作手冊進行問答。\
                                    根據下方的文本資訊回答用戶的問題，只根據文本內容做回答。\
                                    若問題無法從文本得到答案，回答你不知道。"),
            MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template("問題: {query}. \n 文本資訊: {information}"),  # Where the human input will injected
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", 
                                      input_key="query",
                                      return_messages=True)
    
    print('Restart LLM Chain!')
    chain = LLMChain(
        llm=client(),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    return chain

def get_response(query, chroma_collection, chain):

    # query = '有哪些假別以及使用條件?'
    # query = '事假有幾天?'

    results = chroma_collection.query(query_texts=query, n_results=3, include=['documents', 'embeddings'])

    retrieved_documents = results['documents'][0]

    response = chain.predict(query=query, information='\n\n'.join(retrieved_documents))
    
    return response

if __name__ == '__main__':
    # dotenv_path = os.path.dirname(os.getcwd())
    # print(f'dotenv_path: {dotenv_path}')
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    print(f'OPENAI_API_KEY: {OPENAI_API_KEY}')