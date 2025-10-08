from openai import OpenAI

# deploy開
from api.vector_search import vector_search_light

# local開
# from vector_search import (
#     vector_search_light,
# )


# token = os.environ["GITHUB_TOKEN"]
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"


V_SENPAI_SYSTEM_PROMPT = """
你是「V-Senpai」，一位具備豐富經驗的學長姊模擬機器人。你的任務是協助學生了解輔仁大學資管系「系統分析與設計」課程（又稱 SA、小專題）與「專題實作」之間的差異與歷屆經驗。
你會根據歷屆學生的訪談紀錄與課程背景知識，扮演一位中文課堂助教，幫助學生釐清困惑、提供建議與經驗分享。
請嚴格遵守以下規則：
1. **資料為本，禁止猜測或捏造資訊。**  
   - 回答只能根據資料中出現的內容（例如：訪談、課程規劃等）。  
   - 若找不到答案，請說：「我找不到相關資料」，並鼓勵學生改問其他角度。  
2. **問題模糊時，協助釐清再回答。**  
   - 若學生問題不清楚，請主動列出選項或追問，協助對方聚焦。  
3. **回答方式要具體、真誠、有條理。**  
   - 舉例時請指出是來自「某位同學的經驗」。  
   - 不要使用過於空泛的建議，例如「多努力」、「加油就好」這類無實質幫助的回答。  
4. **以中文作答。**  
   - 回答要口語、自然、簡潔明確。
"""


def get_vector_search_result(user_input: str) -> dict:
    """只進行向量搜尋，不回傳 LLM 回應"""
    search_result = vector_search_light(user_input)
    print("🔍 向量搜尋結果:", search_result)

    return {
        "sources": search_result.get("sources", []),
        "ids": search_result.get("ids", []),
        "matches": search_result.get("matches", []),
        "context_text": search_result.get("text", "查無資料。"),
    }


def get_openai_response(token: str, user_input: str, context_text: str = None) -> str:
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=token,
    )

    # 如果沒有提供 context_text，則重新搜尋
    if context_text is None:
        search_result = vector_search_light(user_input)
        context_text = search_result.get("text", "查無資料。")
    messages = [
        {
            "role": "system",
            "content": V_SENPAI_SYSTEM_PROMPT
            + f"\n\n以下是你可以參考的資料：\n{context_text}",
        },
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=1.0, top_p=1.0
    )

    # print("AAA機器人收到的資料",messages)
    print("AAA機器人回應", response.choices[0].message.content)
    return {
        "answer": response.choices[0].message.content,
    }
