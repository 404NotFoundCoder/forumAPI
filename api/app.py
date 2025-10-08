from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()


from api.delete import delete_from_pinecone
from api.llm_client import get_openai_response, get_vector_search_result
from api.upload import upload_to_pinecone

# local 開
# from llm_client import get_groq_response, get_openai_response

app = Flask(__name__)
CORS(app)  # 啟用 CORS


@app.route("/")
def home():
    return jsonify({"message": "LLM Flask API is running."})


@app.route("/api/search", methods=["POST"])
def search():
    """先回傳向量搜尋結果"""
    data = request.get_json()
    user_input = data.get("message", "")

    result = get_vector_search_result(user_input)

    print("✅ 即將回傳搜尋結果：", result)
    return jsonify(result), 200


@app.route("/api/answer", methods=["POST"])
def answer():
    """使用已搜尋的結果來取得 LLM 回應"""
    data = request.get_json()
    user_input = data.get("message", "")
    access_token = data.get("accessToken")
    context_text = data.get("context_text", None)

    result = get_openai_response(access_token, user_input, context_text)

    print("✅ 即將回傳答案：", result)
    return jsonify(result), 200


@app.route("/api/test", methods=["POST"])
def test():
    """原本的完整流程（向後相容）"""
    data = request.get_json()
    # print(f"收到的資料: {data}")

    user_input = data.get("message", "")
    access_token = data.get("accessToken")

    result = get_openai_response(access_token, user_input)

    print("✅ 即將回傳資料：", result)
    return jsonify(result), 200


@app.route("/api/upload", methods=["POST"])
def upload():
    """
    上傳文本內容到 Pinecone 向量數據庫
    接收 JSON 格式：
    {
        "id": "唯一識別碼",
        "source": "資料來源",
        "content": "要上傳的文本內容"
    }
    """
    try:
        data = request.get_json()

        # 驗證必要欄位
        if not data:
            return jsonify({"error": "請提供 JSON 資料"}), 400

        id = data.get("id")
        source = data.get("source")
        content = data.get("content")

        if not all([id, source, content]):
            return (
                jsonify(
                    {"error": "缺少必要欄位", "required": ["id", "source", "content"]}
                ),
                400,
            )

        # 調用上傳函數
        upload_to_pinecone(id, source, content)

        return jsonify({"message": "上傳成功", "id": id, "source": source}), 200

    except Exception as e:
        print(f"❌ 上傳 API 錯誤：{e}")
        return jsonify({"error": "上傳失敗", "details": str(e)}), 500


@app.route("/api/delete", methods=["DELETE"])
def delete():
    """
    從 Pinecone 向量數據庫中刪除指定 ID 的資料
    接收 JSON 格式：
    {
        "id": "要刪除的向量 ID"
    }
    """
    try:
        data = request.get_json()

        # 驗證必要欄位
        if not data:
            return jsonify({"error": "請提供 JSON 資料"}), 400

        id = data.get("id")

        if not id:
            return jsonify({"error": "缺少必要欄位", "required": ["id"]}), 400

        # 調用刪除函數
        success = delete_from_pinecone(id)

        if success:
            return jsonify({"message": "刪除成功", "id": id}), 200
        else:
            return jsonify({"error": "刪除失敗", "id": id}), 500

    except Exception as e:
        print(f"❌ 刪除 API 錯誤：{e}")
        return jsonify({"error": "刪除失敗", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
