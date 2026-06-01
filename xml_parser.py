
import requests
from supabase import create_client, Client
import time
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning

# Отключаем предупреждение парсера
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ================= НАСТРОЙКИ =================
XML_FILE = "democracykz-.WordPress.2026-05-31.xml"

# ВПИШИ СВОИ КЛЮЧИ СЮДА:
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_html(raw_html):
    if not raw_html: return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def get_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-004:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "model": "models/gemini-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_DOCUMENT"
    }
    
    max_retries = 5 # Сколько раз будем пытаться пробить лимит
    
    for attempt in range(max_retries):
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()['embedding']['values']
        elif response.status_code == 429:
            print(f"⏳ Перегрев токенов (429)! Ждем 15 сек... (Попытка {attempt + 1}/{max_retries})")
            time.sleep(15) # Ждем, чтобы минутный лимит Гугла успел обнулиться
        else:
            print(f"❌ Ошибка Gemini: {response.text}")
            return None
            
    print("❌ Статья слишком огромная или лимит намертво заблокирован. Пропускаем.")
    return None

# ================= ОСНОВНОЙ ЦИКЛ =================
def run_xml_parser():
    print(f"🚀 Читаем локальный файл {XML_FILE}...")
    
    try:
        with open(XML_FILE, "r", encoding="utf-8") as file:
            xml_data = file.read()
    except FileNotFoundError:
        print(f"❌ Файл {XML_FILE} не найден в папке!")
        return

    soup = BeautifulSoup(xml_data, "html.parser")
    items = soup.find_all("item")
    
    print(f"📂 Найдено записей в файле: {len(items)}")
    
    total_added = 0

    for item in items:
        post_type_tag = item.find("wp:post_type")
        status_tag = item.find("wp:status")
        
        post_type = post_type_tag.text if post_type_tag else ""
        status = status_tag.text if status_tag else ""
        
        if post_type != "post" or status != "publish":
            continue

        # Извлекаем данные
        post_id_tag = item.find("wp:post_id")
        title_tag = item.find("title")
        content_tag = item.find("content:encoded")
        link_tag = item.find("link")

        # Получаем ID статьи, если его нет — ставим 0, чтобы база не ругалась
        post_id = int(post_id_tag.text) if post_id_tag and post_id_tag.text.isdigit() else 0
        title = clean_html(title_tag.text) if title_tag else "Без заголовка"
        raw_content = content_tag.text if content_tag else ""
        content = clean_html(raw_content)
        link = link_tag.text if link_tag else ""

        if not content or len(content) < 50:
            continue

        print(f"Векторизуем [ID: {post_id}]: {title[:40]}...")
        
        embedding = get_embedding(title + ". " + content)
        
        if embedding:
            try:
                supabase.table("democracy_news").insert({
                    "post_id": post_id, # ТЕПЕРЬ ПЕРЕДАЕМ ID
                    "title": title,
                    "content": content,
                    "url": link,
                    "embedding": embedding
                }).execute()
                
                print("✅ Сохранено!")
                total_added += 1
            except Exception as e:
                if '23505' in str(e):
                    print("⚠️ Статья уже существует, пропускаем.")
                else:
                    print(f"❌ Ошибка БД: {e}")
        
        time.sleep(1) 

    print(f"\n🎉 Готово! Всего загружено уникальных статей: {total_added}")

if __name__ == "__main__":
    run_xml_parser()