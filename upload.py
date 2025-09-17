import json
import os

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# 載入 .env
load_dotenv()


def upload_file(file_path):
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

    # 讀取 JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = []
    co = cohere.Client(cohere_api_key)

    for item in data:
        question = item["metadata"].get("question", "")
        answer = item["metadata"].get("answer", "")
        text = question + "\n" + answer

        print(f"🔍 正在處理 ID: {item['id']}，文本內容: {text}")

        embedding = co.embed(
            texts=[text], model="embed-multilingual-v3.0", input_type="search_document"
        ).embeddings[0]

        vectors.append(
            {"id": item["id"], "values": embedding, "metadata": item["metadata"]}
        )

    try:
        index.upsert(vectors=vectors)
        print(f"✅ 成功上傳 {len(vectors)} 筆向量到 Pinecone index `{index_name}`！")
    except Exception as e:
        print(f"❌ 上傳失敗：{e}")


if __name__ == "__main__":
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("vec-0601-bk")

    # 抓一筆 (例如 doc1_q03)
    res = index.fetch(
        ids=["interview_srsd_issues"], namespace=""
    )  # namespace 沒設定就用預設 ""
    print(res)
