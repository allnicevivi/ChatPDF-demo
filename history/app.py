
from service import Retriever, ChainUtils

import streamlit as st
import pprint as pp

def main():

    st.title("ChatPDF DEMO")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    # print(f'uploaded_file: {uploaded_file}')
    # pp.pprint(f'st.session_state: {st.session_state}')

    if 'chroma_collection' not in st.session_state:
        st.session_state.chroma_collection = None
        st.session_state.chain = None

    if uploaded_file is not None and st.session_state.chroma_collection is None:

        chroma_collection = Retriever.getRetriver(uploaded_file).main()

        chain = ChainUtils.llm_chain()

        st.session_state.chroma_collection = chroma_collection
        st.session_state.chain = chain

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

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

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            resp = ChainUtils.get_response(prompt, st.session_state.chroma_collection, st.session_state.chain)
            response = st.write(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == '__main__':
    main()