from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from supabase import create_client, Client

app = FastAPI()

# Разрешаем виджету с твоего сайта обращаться к этому API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # В будущем можно заменить на ["https://democracy.kz"] для максимальной безопасности
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ключи будут подтягиваться из защищенных переменных сервера
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat(request: QueryRequest):
    user_query = request.query
    if not user_query:
        raise HTTPException(status_code=400, detail="Пустой запрос")

    try:
        # 1. Векторизуем вопрос пользователя (указываем тип задачи RETRIEVAL_QUERY)
        embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
        embed_payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": user_query}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        embed_res = requests.post(embed_url, json=embed_payload).json()
        query_embedding = embed_res['embedding']['values']

        # 2. Ищем похожие статьи в Supabase
        match_res = supabase.rpc('match_news', {
            'query_embedding': query_embedding,
            'match_threshold': 0.4, # Степень строгости поиска
            'match_count': 3        # Берем топ-3 статьи
        }).execute()
        
        matched_news = match_res.data

        # 3. Собираем контекст из найденных текстов
        context_text = "Вот найденные статьи с сайта:\n\n"
        if matched_news:
            for news in matched_news:
                context_text += f"Заголовок: {news['title']}\nТекст: {news['content']}\nСсылка: {news['url']}\n\n"
        else:
            context_text += "К сожалению, релевантных статей не найдено.\n\n"

        # 4. Генерируем ответ через Gemini
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
        prompt = f"""Ты — вежливый AI-консультант новостного портала Democracy.kz. 
Твоя задача — компетентно ответить на вопрос пользователя, используя ТОЛЬКО предоставленный ниже контекст из наших статей. 
Если в текстах статей нет ответа на вопрос, честно скажи, что пока такой информации на сайте нет, и категорически не придумывай ничего от себя. 
Отвечай на том же языке, на котором задан вопрос. Если берешь информацию из статьи, в конце ответа обязательно прикрепи ссылку на неё.

Контекст (Статьи с портала):
{context_text}

Вопрос пользователя: {user_query}"""

        gen_payload = {"contents": [{"parts": [{"text": prompt}]}]}
        gen_res = requests.post(gen_url, json=gen_payload).json()
        
        answer = gen_res['candidates'][0]['content']['parts'][0]['text']
        
        return {"answer": answer}

    except Exception as e:
        print(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")