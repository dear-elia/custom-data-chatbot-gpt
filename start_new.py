import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader

openai.api_key = 'YOUR API KEY';
st.header("Chat with the Pai Chai chatbot 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="C:/Users/PCU/VScode_project/data", recursive=True)
        docs = reader.load_data()
        
        # Set up the configuration directly in VectorStoreIndex if Settings cannot be directly instantiated
        index = VectorStoreIndex.from_documents(
            docs, 
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
                       system_prompt="당신은 대학의 PDF 자료를 기반으로 사용자 질문에 정확하게 답변하는 능숙한 대학 조교입니다. "
                                      "사용자가 한국어로 질문하면 항상 한국어로 답변하고, 영어로 질문하면 영어로 답변해야 합니다. "
                                      "정확하고 간결하며 사용자의 질문에 맞는 맥락 중심의 답변을 제공하는 것이 목표입니다. "
                                      "충분한 맥락이 주어진 경우 해당 맥락을 바탕으로 답변하고, 부족할 경우 일반적인 지식을 활용하여 응답하세요. "
                                      "맥락을 충분히 반영할 수 없는 경우, 답변에 일반적인 지식을 바탕으로 하되, "
                                      "사용자가 정확한 정보를 확인할 수 있도록 공식 출처를 권장하십시오. "
                                      "가능하다면 세 문장 내로 간결하게 답변을 제공하세요.")
        )
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

