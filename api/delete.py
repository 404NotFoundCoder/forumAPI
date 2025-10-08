import os

from dotenv import load_dotenv
from pinecone import Pinecone

# 載入 .env
load_dotenv()


def delete_from_pinecone(id):
    """
    從 Pinecone 向量數據庫中刪除指定 ID 的資料

    Args:
        id (str): 要刪除的向量 ID

    Returns:
        bool: 刪除成功返回 True，失敗返回 False
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_api_key:
        print("❌ 請確認 .env 中是否正確設定 PINECONE_API_KEY")
        return False

    try:
        # 初始化 Pinecone
        pc = Pinecone(api_key=pinecone_api_key)

        index_name = "vec-0601-bk"

        # 檢查 index 是否存在
        if index_name not in pc.list_indexes().names():
            print(f"❌ Pinecone index `{index_name}` 不存在")
            return False

        index = pc.Index(index_name)

        print(f"🗑️ 正在刪除 ID: {id}")

        # 刪除指定 ID 的向量
        index.delete(ids=[id])

        print(f"✅ 成功刪除 ID: {id}")
        return True

    except Exception as e:
        print(f"❌ 刪除失敗：{e}")
        return False
