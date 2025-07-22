from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_model():
    """
    HuggingFace 임베딩 모델을 반환합니다.
    """
    embeddings_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
    return embeddings_model