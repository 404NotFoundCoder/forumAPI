import os
import sys

import cohere
from dotenv import load_dotenv
from pinecone import Pinecone

# 設定標準輸出編碼為 UTF-8
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# 初始化 Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "vec-0601"
index = pc.Index(index_name)

# 初始化 Cohere
co = cohere.ClientV2()


def vector_search_light(user_input: str) -> dict:
    try:
        response = co.embed(
            texts=[user_input],
            model="embed-multilingual-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )

        # 取得向量
        vector = response.embeddings.float_[0]

        # 查詢 Pinecone
        results = index.query(
            # namespace="interview-rag",
            vector=vector,
            top_k=3,
            include_values=False,
            include_metadata=True,
        )

        matches = results.get("matches", [])

        # 補一個合併所有文字的欄位（給 Prompt 用）
        formatted = "\n\n---\n\n".join(
            f"{match['metadata']['interviewee']}說：\nQ：{match['metadata']['question']}\nA：{match['metadata']['answer']}"
            for match in matches
        )
        sources = [match["metadata"]["source"] for match in matches]

        print(f"🔍 向量查詢結果數量: {len(matches)}")
        # print(f"🔍 向量查詢結果內容: {formatted[:200]}...")  # 只顯示前200個字

        return {
            "matches": matches,
            "sources": sources,
            "text": formatted,  # ✅ 給 prompt 直接使用
            "usage": results.get("usage", {}),
        }
    except Exception as e:
        print(f"❌ 向量查詢錯誤: {str(e)}")
        return {"matches": [], "text": "查無資料。", "usage": {}, "error": str(e)}


if __name__ == "__main__":
    vector_search_light("我們最後專題要用英文報告嗎？")
