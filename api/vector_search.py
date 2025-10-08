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
index_name = "vec-0601-bk"
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
            top_k=50,
            include_values=False,
            include_metadata=True,
        )

        # 篩選 score > 0.5 的結果
        all_matches = results.get("matches", [])
        filtered_matches = [
            match for match in all_matches if match.get("score", 0) > 0.5
        ]
        # 依 score 由高到低排序
        filtered_matches = sorted(
            filtered_matches, key=lambda x: x.get("score", 0), reverse=True
        )

        # 取篩選後最高的三筆（可能少於三筆）
        top_three_matches = filtered_matches[:3]

        # 補一個合併所有文字的欄位（給 Prompt 用）- 使用篩選後最高的三筆
        formatted = "\n\n---\n\n".join(
            f"Q：{match['metadata']['source']}\nA：{match['metadata']['content']}"
            for match in top_three_matches
        )
        sources = [match["metadata"]["source"] for match in top_three_matches]
        ids = [m["id"] for m in top_three_matches]

        print(f"🔍 向量查詢結果數量: {len(filtered_matches)}")
        print(f"🔍 篩選後最高三筆數量: {len(top_three_matches)}")
        # print(f"🔍 向量查詢結果內容: {formatted[:200]}...")  # 只顯示前200個字

        # 將 matches 轉換為可序列化的格式
        serializable_matches = []
        for match in filtered_matches:
            serializable_match = {
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": match.get("metadata", {}),
            }
            serializable_matches.append(serializable_match)

        return {
            "matches": serializable_matches,
            "sources": sources,
            "ids": ids,
            "text": formatted,  # ✅ 給 prompt 直接使用
            "usage": results.get("usage", {}),
        }
    except Exception as e:
        print(f"❌ 向量查詢錯誤: {str(e)}")
        return {"matches": [], "text": "查無資料。", "usage": {}, "error": str(e)}


if __name__ == "__main__":
    vector_search_light("我們最後專題要用英文報告嗎？")
