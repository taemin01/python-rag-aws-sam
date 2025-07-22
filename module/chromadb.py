import chromadb
from chromadb.config import Settings

def get_chromadb_collection(json_file):
    """
    ChromaDB 컬렉션을 반환합니다.
    컬렉션 이름은 파일 경로 기반으로 생성됩니다.
    """
    load_dotenv('../RAG_Pipeline/.env')

    client = chromadb.HttpClient(
            host=os.getenv('CHROMADB_HOST'),
            port=os.getenv('CHROMADB_PORT'),
            settings=Settings(anonymized_telemetry=False)
        )

    # JSON 파일 이름을 기반으로 컬렉션 이름 생성
    collection_name = f'document_{os.path.basename(json_file)[0]}'
    print(f"ChromaDB 컬렉션 이름: {collection_name}")

    collection = client.get_or_create_collection(name=collection_name)
    return collection