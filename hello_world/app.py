import json
import requests
import os
import chromadb
import httpx
from chromadb.config import Settings
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )

client = chromadb.HttpClient(
            host=os.getenv('CHROMADB_HOST'),
            port=os.getenv('CHROMADB_PORT'),
            settings=Settings(anonymized_telemetry=False)
        )
OPENAI_API_KEY = os.getenv('OPENAI_KEY')

def lambda_handler(event, context):
    request_body = json.loads(event)
    request_collection_name = request_body.get('collection_name')
    request_user_query = request_body.get('user_query')

    if not request_collection_name:
        return {
            'statusCode': 400,
            'body': json.dumps({'error' : 'Missing "collection_name" in request body'})
        }
    
    collection_name = f'document_{request_collection_name}' # 중괄호 안에 입력받은 파일 A~H까지 입력 받게끔 해야함 POST로
    print(f"ChromaDB 컬렉션 이름: {collection_name}")
    print(f"사용자 질문 : {request_user_query}\n")

    user_query_embedding = embeddings_model.embed_qurey(request_user_query)

    results = collection_name.query(
        query_embeddings=[user_query_embedding],
        n_result=10,
        include=['documents', 'metadatas']
    )

    retriever_list = []
    if results and results['documents'] and results['metadatas']:
        for doc, metadata in zip(results['documents'], results['metadatas']):
            doc_embedding = embeddings_model.embed_query(doc)

            distance = np.linalg.norm(np.array(user_query_embedding) - np.array(doc_embedding))

            start_page = metadata.get('start_page', 'N/A')
            end_page = metadata.get('end_page', 'N/A')
            
            retriever_list.append(f"페이지 : {start_page} - {end_page}: \n 유사도 : {distance} \n {doc}\n")

    context_text = "\n".join(retriever_list)
    
    prompt = f"""Question: {request_user_query}

    Reference Context:
    {context_text}

    Instructions:
    1. 제공된 참고 문맥만을 기반으로 답변하세요
    2. 외부 출처나 일반 지식을 포함하지 마세요
    3. 문맥에서 관련 정보를 찾을 수 없다면 "제공된 문맥에는 이에 대한 정보가 없습니다"라고 답변하세요
    4. 답변에서 항상 문맥의 특정 부분을 인용하세요. 인용 시 문서 이름과 페이지 범위를 명확히 언급하세요 (예: "참고문서 A 한화생명 간편가입 암보험.json (페이지: 10~12)에 따르면...")
    5. 확실하지 않은 경우 불확실성을 인정하세요

    Response Format:
    1. 직접적인 답변
    2. 참고 문서 이름 및 페이지 범위 (문맥에 없을 시 포함 X)

    Remember: 제공된 맥락에서 제공된 정보만 사용해야 합니다. 참고 문서 이름과 페이지 명시는 필수입니다. 답변은 보기 좋게 줄바꿈 해주세요
    """

    base_url = "https://api.upstage.ai/v1/chat/completions"
    payload = {
        "model": "solar-pro 2",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # timeout -> context가 길어 5초로는 부족, 60초로 늘려줌
    with httpx.Client(timeout=60.0) as client:
        response = client.post(base_url, headers=headers, json=payload)
    
    result = response.json()
    llm_result = result['choices'][0]['message']['content']

    return {
        "statusCode": 200,
        "body": json.dumps({
            "user_query" : request_user_query,
            "results" : llm_result
        }),
    }