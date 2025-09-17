import json
import os

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 載入 .env
load_dotenv()

# source = question+(interviewee)
# CONTENT = 根據interviewee的經驗，answer


def upload_to_pinecone(id, source, content):
    cohere_api_key = os.getenv("CO_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not cohere_api_key or not pinecone_api_key:
        print("❌ 請確認 .env 中是否正確設定 COHERE_API_KEY 與 PINECONE_API_KEY")
        return

    # 初始化 Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "vec-0601-bk"

    if index_name not in pc.list_indexes().names():
        print(f"⚙️ 建立新的 Pinecone index: {index_name} (serverless)...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    vectors = []
    co = cohere.Client(cohere_api_key)

    print(f"🔍 正在處理 ID: {id}，文本內容: {content}")

    embedding = co.embed(
        texts=[content], model="embed-multilingual-v3.0", input_type="search_document"
    ).embeddings[0]

    vectors.append(
        {
            "id": id,
            "values": embedding,
            "metadata": {
                "source": source,
                "content": content,
            },
        }
    )

    try:
        index.upsert(vectors=vectors)
        print(f"✅ 成功上傳 {len(vectors)} 筆向量到 Pinecone index `{index_name}`！")
    except Exception as e:
        print(f"❌ 上傳失敗：{e}")
