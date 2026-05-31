"""
Democracy.kz AI Chatbot — Рефакторинг Backend
FastAPI + Supabase RAG + Gemini Streaming

Изменения:
- Асинхронный стриминг через StreamingResponse (token-by-token)
- Полноценный Error Handling с вежливыми сообщениями
- Улучшенный prompt с Markdown-форматированием
- Rate Limiting (in-memory, per IP, 15 req/min)
- Полностью async (httpx вместо requests)
"""

import os
import json
import time
import asyncio
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from supabase import create_client, Client

# ─────────────────────────────────────────────
# Инициализация приложения
# ─────────────────────────────────────────────

app = FastAPI(title="Democracy.kz AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    # В продакшене замени на ["https://democracy.kz"]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Конфигурация из переменных окружения
# ─────────────────────────────────────────────

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise RuntimeError(
        "Отсутствуют обязательные переменные окружения: "
        "SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
# Rate Limiter (in-memory, per IP)
# ─────────────────────────────────────────────

RATE_LIMIT_REQUESTS = 15   # максимум запросов
RATE_LIMIT_WINDOW = 60     # за это кол-во секунд

# Структура: { ip: [timestamp1, timestamp2, ...] }
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> None:
    """
    Проверяет, не превысил ли IP лимит запросов.
    Поднимает HTTP 429, если лимит исчерпан.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Оставляем только запросы внутри текущего окна
    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip] if ts > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                "Вы отправляете слишком много запросов. "
                f"Пожалуйста, подождите {RATE_LIMIT_WINDOW} секунд."
            ),
        )

    _rate_limit_store[client_ip].append(now)


# ─────────────────────────────────────────────
# Pydantic-модели
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


# ─────────────────────────────────────────────
# Системный промпт
# ─────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """Ты — вежливый и профессиональный AI-консультант новостного портала Democracy.kz.

**Правила работы:**
1. Отвечай ТОЛЬКО на основе предоставленного контекста из статей сайта.
2. Если ответа в контексте нет — честно сообщи: "К сожалению, на портале пока нет информации по этому вопросу."
3. Никогда не придумывай факты, даты, имена или события.
4. Отвечай на том же языке, на котором задан вопрос (казахский, русский, английский).

**Формат ответа (обязательно используй Markdown):**
- Используй **жирный текст** для ключевых терминов и важных фактов.
- Используй маркированные списки (`-`) для перечислений из нескольких пунктов.
- Используй нумерованные списки (`1. 2. 3.`) для последовательных шагов.
- Если упоминаешь статью — оформляй ссылку так: [Название статьи](URL)
- В конце ответа добавляй раздел **📚 Источники:** со списком использованных ссылок.
- Не используй заголовки H1 (`#`) — только H3 (`###`) или меньше.
- Длина ответа: компактно, по делу. Без воды.

**Контекст (статьи с портала Democracy.kz):**
{context}

**Вопрос пользователя:** {query}"""


# ─────────────────────────────────────────────
# Вспомогательные async-функции
# ─────────────────────────────────────────────

async def get_query_embedding(client: httpx.AsyncClient, query: str) -> list[float]:
    """Векторизует запрос пользователя через Gemini Embedding API."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-embedding-001:embedContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": query}]},
        "taskType": "RETRIEVAL_QUERY",
    }
    response = await client.post(url, json=payload, timeout=15.0)
    response.raise_for_status()
    data = response.json()

    try:
        return data["embedding"]["values"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Неожиданный ответ от Embedding API: {data}") from exc


async def search_supabase(embedding: list[float]) -> list[dict]:
    """Выполняет векторный поиск по Supabase."""
    # Supabase Python SDK синхронный — запускаем в executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: supabase.rpc(
            "match_news",
            {
                "query_embedding": embedding,
                "match_threshold": 0.4,
                "match_count": 3,
            },
        ).execute(),
    )
    return result.data or []


def build_context(matched_news: list[dict]) -> str:
    """Формирует текстовый контекст из найденных статей."""
    if not matched_news:
        return "Релевантных статей по данному запросу не найдено."

    parts = []
    for i, news in enumerate(matched_news, start=1):
        title = news.get("title", "Без заголовка")
        content = news.get("content", "")[:1500]  # ограничиваем размер
        url = news.get("url", "")
        parts.append(
            f"**Статья {i}: {title}**\n"
            f"URL: {url}\n"
            f"Содержание: {content}\n"
        )
    return "\n---\n".join(parts)


async def stream_gemini_response(prompt: str):
    """
    Генератор, который стримит ответ от Gemini token-by-token.
    Используется как источник для StreamingResponse.
    Данные отправляются в формате Server-Sent Events (SSE).
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:streamGenerateContent"
        f"?key={GEMINI_API_KEY}&alt=sse"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,      # низкая температура = меньше галлюцинаций
            "maxOutputTokens": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # SSE-формат: строки начинаются с "data: "
                    if not line.startswith("data:"):
                        continue

                    raw = line[len("data:"):].strip()

                    # Gemini сигнализирует о конце потока
                    if raw == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw)
                        text = (
                            chunk["candidates"][0]["content"]["parts"][0]["text"]
                        )
                        # Отправляем каждый чанк как SSE-событие
                        yield f"data: {json.dumps({'token': text})}\n\n"

                    except (KeyError, IndexError, json.JSONDecodeError):
                        # Пропускаем служебные чанки без текста
                        continue

    except httpx.HTTPStatusError as exc:
        error_msg = f"Ошибка Gemini API (HTTP {exc.response.status_code})."
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

    except httpx.RequestError:
        yield f"data: {json.dumps({'error': 'Не удалось подключиться к Gemini API. Попробуйте позже.'})}\n\n"

    # Финальный сигнал — клиент закрывает соединение
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────
# Эндпоинты
# ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: QueryRequest, http_request: Request):
    """
    Основной эндпоинт чата.
    Возвращает StreamingResponse в формате Server-Sent Events.
    """
    # 1. Rate Limiting
    client_ip = http_request.client.host if http_request.client else "unknown"
    check_rate_limit(client_ip)

    # 2. Валидация запроса
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Запрос не может быть пустым.")
    if len(user_query) > 1000:
        raise HTTPException(status_code=400, detail="Запрос слишком длинный (максимум 1000 символов).")

    # 3. Получаем эмбеддинг + контекст из Supabase
    try:
        async with httpx.AsyncClient() as client:
            embedding = await get_query_embedding(client, user_query)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка сервиса эмбеддингов (HTTP {exc.response.status_code}). Попробуйте позже.",
        )
    except (ValueError, httpx.RequestError) as exc:
        raise HTTPException(status_code=502, detail=f"Не удалось векторизовать запрос: {exc}")

    try:
        matched_news = await search_supabase(embedding)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка поиска по базе данных: {exc}",
        )

    # 4. Строим промпт
    context = build_context(matched_news)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, query=user_query)

    # 5. Стримим ответ клиенту
    return StreamingResponse(
        stream_gemini_response(prompt),
        media_type="text/event-stream",
        headers={
            # Отключаем буферизацию на промежуточных прокси
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health_check():
    """Простой health-check для мониторинга."""
    return {"status": "ok", "version": "2.0.0"}
