from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader

app = Flask(__name__)

#openai.api_key = 'API_KEY';

def load_data():
    reader = SimpleDirectoryReader(input_dir="C:/Users/PCU/VScode_project/data", recursive=True)
    docs = reader.load_data()
        
    # Set up the configuration directly in VectorStoreIndex if Settings cannot be directly instantiated
    index = VectorStoreIndex.from_documents(
        docs, 
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, 
                   system_prompt="당신은 대학의 PDF 자료를 기반으로 사용자 질문에 정확하게 답변하는 능숙한 대학 조교입니다. "
                                      "항상 한국어로만 대답하고 영어를 사용하지 마세요. "
                                      "사용자의 질문이 다른 언어로 되어 있어도 반드시 한국어로 응답하십시오. "
                                      "정확하고 간결하며 사용자의 질문에 맞는 맥락 중심의 답변을 제공하는 것이 목표입니다. "
                                      "충분한 맥락이 주어진 경우 해당 맥락을 바탕으로 답변하고, 부족할 경우 일반적인 지식을 활용하여 응답하세요. "
                                      "맥락을 충분히 반영할 수 없는 경우, 답변에 일반적인 지식을 바탕으로 하되, "
                                      "사용자가 정확한 정보를 확인할 수 있도록 공식 출처를 권장하십시오. "
                                      "가능하다면 세 문장 내로 간결하게 답변을 제공하세요.")
    )
    return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Flask route to handle chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('message', '')

    if not user_prompt:
        return jsonify({"error": "No message provided"}), 400

    # Process the prompt through the chat engine
    response = chat_engine.chat(user_prompt)
    assistant_response = response.response

    return jsonify({"response": assistant_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

