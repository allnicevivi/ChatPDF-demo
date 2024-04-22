from src import GenerationManager, RetrieveManager

import streamlit as st
import pandas as pd
from datetime import datetime
import openai
from llama_index.core import Settings
# Get the current time in UTC+8 timezone
import nest_asyncio
nest_asyncio.apply()
import pytz
utc8_timezone = pytz.timezone('Asia/Shanghai')

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
# openai.api_key = st.secrets.openai_key
# df_path = f'/content/drive/MyDrive/Vivi/LLM/data/員工工作手冊/demo_QA_History.xlsx'
# st.session_state.df = pd.read_excel(df_path)

st.title("ChatDoc DEMO")

with st.sidebar:
    flow_version = st.radio(
        "Choose the version.",
        ("V1", "V2 (Sub-Questions)", "V3 (Gen-Questions)")
    )
    flow_version = flow_version[0:2]

    option_llm = st.selectbox(
        "Choose the LLM model.",
        ("gpt-35-turbo-16k", "gpt-4", "Breeze"),
        index=None,
        placeholder="Select demo file",
    )

    uploaded_file = st.selectbox(
        "Choose the demo file.",
        ("新向系統", "永純化學", "捷智商訊"),
        index=None,
        placeholder="Select demo file",
    )


# # @st.cache_resource(show_spinner=False)
# def load_data(uploaded_file):
#     with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
#         return RetrieveManager.get_retriever(uploaded_file)

# # @st.cache_resource(show_spinner=False)
# def load_llm(option_llm):
#     with st.spinner(text="Loading the LLM model! This should take 1-2 minutes."):
#         return GenerationManager.get_LLM(option_llm)

if option_llm != None and uploaded_file != None:
    if ("option_llm" not in st.session_state.keys()) or (option_llm != st.session_state.option_llm):
        st.session_state.option_llm = option_llm
        # load_llm.clear()
        print(option_llm)
        # if option_llm == None:
        #     option_llm = "gpt-3.5-turbo"
        st.session_state.llm = GenerationManager.get_LLM(option_llm)
        st.session_state.update_engine = True
        Settings.chunk_size = 512
        Settings.llm = st.session_state.llm
    if ("uploaded_file" not in st.session_state.keys()) or (uploaded_file != st.session_state.uploaded_file):
        st.session_state.uploaded_file = uploaded_file
        # load_data.clear()
        nodes, st.session_state.index = RetrieveManager.get_retriever(uploaded_file, st.session_state.llm)
        st.session_state.update_engine = True

    if (st.session_state.update_engine == True) or (flow_version != st.session_state['flow_version']):
        query_engine, sub_query_engine = GenerationManager.get_engine(st.session_state.index, st.session_state.llm)
        if flow_version == 'V1':
            st.session_state.chat_engine = query_engine
        else:
            st.session_state.chat_engine = sub_query_engine
        st.session_state.update_engine = False

st.session_state['flow_version'] = flow_version
# # Initialize chat history
# if "messages" not in st.session_state:
#     # st.session_state.messages = []
#     st.session_state.messages = [
#         {"role": "assistant", "content": "Ask me a question about the file!"}
#     ]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Say something"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    start_dt = datetime.now(utc8_timezone)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if st.session_state['flow_version'] == 'V1':
            response = GenerationManager.get_response(st.session_state.chat_engine, prompt)
            response_str = response.response
            retrieved_doc = ''
            subQA = ''
            for i in range(len(response.source_nodes)):
                cont = response.source_nodes[i].node.get_content()
                retrieved_doc += cont + '\n\n'
        if st.session_state['flow_version'] == 'V2':
            response = GenerationManager.get_response(st.session_state.chat_engine, prompt)
            response_str = response.response
            retrieved_doc = ''
            subQA = ''
            for i in range(len(response.source_nodes)):
                cont = response.source_nodes[i].node.get_content()
                if 'Sub question' in cont:
                    subQA += cont + '\n\n'
                else:
                    retrieved_doc += cont + '\n\n'
        elif st.session_state['flow_version'] == 'V3':
            response_str, gen_queries, gen_response = GenerationManager.rewrite_and_get_response(st.session_state.llm, st.session_state.chat_engine, prompt)
            retrieved_doc = ''
            subQA = ''

            for i, gen_query in enumerate(gen_queries):
                subQA += f'Sub question: {gen_query}?\n Response: {gen_response[i]}\n\n'

        time_spent = datetime.now(utc8_timezone) - start_dt
        # st.session_state.df = st.session_state.df.append({'Datetime': datetime.now(utc8_timezone).strftime("%Y-%m-%d %H:%M:%S"),
        #                                                   'Time Spent': round(time_spent.total_seconds(), 2),
        #                                                   'Version': st.session_state['flow_version'],
        #                                                   'File': st.session_state.uploaded_file,
        #                                                   'LLM': st.session_state.option_llm,
        #                                                   'Query': prompt,
        #                                                   'Sub-QA': subQA,
        #                                                   'Retrieve': retrieved_doc,
        #                                                   'Response': response_str}, ignore_index=True)
        # st.session_state.df.to_excel(df_path, index=False)
        st.write(response_str)
    st.session_state.messages.append({"role": "assistant", "content": response_str})




# # If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.write(response.response)
#             message = {"role": "assistant", "content": response.response}
#             st.session_state.messages.append(message) # Add response to message history