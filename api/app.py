from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()


from api.llm_client import get_openai_response

# local 開
# from llm_client import get_groq_response, get_openai_response

app = Flask(__name__)
CORS(app)  # 啟用 CORS


@app.route("/")
def home():
    return jsonify({"message": "LLM Flask API is running."})


@app.route("/api/test", methods=["POST"])
def test():
    data = request.get_json()
    # print(f"收到的資料: {data}")

    user_input = data.get("message", "")
    access_token = data.get("accessToken")

    result = get_openai_response(access_token, user_input)

    print("✅ 即將回傳資料：", result)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(debug=True)
