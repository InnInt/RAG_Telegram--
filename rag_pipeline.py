# rag_pipeline.py
import os
import pickle
import sqlite3
import faiss
import tiktoken
import hashlib
import requests
import numpy as np
from datetime import datetime
from openai import OpenAI, PermissionDeniedError, AuthenticationError, RateLimitError
from config import FAISS_INDEX_PATH, OPENAI_API_KEY, EMBEDDING_MODEL, GPT_MODEL as CHAT_MODEL, DB_PATH

# Инициализация клиента OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# === Актуальные цены (OpenAI, июль 2025) ===
MODEL_PRICES = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.0100},
    "gpt-4.5": {"prompt": 0.0050, "completion": 0.0200},
    "o1-pro": {"prompt": 0.0150, "completion": 0.0600},
    "o1-mini": {"prompt": 0.0030, "completion": 0.0120}
}

# Используем цены по умолчанию для gpt-4o-mini, если модель не найдена
current_model_prices = MODEL_PRICES.get(CHAT_MODEL, MODEL_PRICES["gpt-4o-mini"])

# === Подсчёт токенов ===
def count_tokens(text, model=CHAT_MODEL):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100kbase")
    return len(enc.encode(text))

# === Хэширование запроса для кэширования ===
def hash_query(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()

# === Инициализация базы данных (логи запросов) ===
def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            response TEXT,
            tokens_prompt INTEGER,
            tokens_completion INTEGER,
            cost_usd REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# === Проверка кэша ===
def get_cached_response(query: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT response FROM query_logs WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

# === Логирование нового запроса ===
def log_query(query: str, response: str, tokens_prompt: int, tokens_completion: int, cost_usd: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT OR REPLACE INTO query_logs 
            (query, response, tokens_prompt, tokens_completion, cost_usd, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, response, tokens_prompt, tokens_completion, cost_usd, datetime.now().isoformat()))
        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Ошибка при логировании: {e}")
    finally:
        conn.close()

# === Загрузка FAISS-индекса и чанков ===
def load_faiss_index():
    index_path = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    pkl_path = os.path.join(FAISS_INDEX_PATH, "index.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS индекс не найден: {index_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Файл чанков не найден: {pkl_path}")

    index = faiss.read_index(index_path)
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# === Получение эмбеддингов через OpenAI ===
def embed_query(query: str):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding
    except (PermissionDeniedError, AuthenticationError, RateLimitError) as e:
        print(f"❌ OpenAI ошибка: {e}")
        raise
    except Exception as e:
        print(f"❌ Неизвестная ошибка при получении эмбеддинга: {e}")
        raise

# === Поиск релевантных чанков ===
def search_chunks(query: str, index, chunks, top_k=5):
    # Получаем эмбеддинг и преобразуем в numpy array с нужным типом
    query_embedding = embed_query(query)
    query_vector = np.array([query_embedding], dtype="float32")  # ← ключевая строка
    D, I = index.search(query_vector, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# === Генерация ответа с источниками ===
def generate_answer(query: str, context_chunks: list):
    # Формируем контекст с источниками
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        truncated_chunk = chunk.strip()[:1000]  # Ограничиваем длину
        context_text += f"[Источник {i+1}]\n{truncated_chunk}\n\n"

    prompt = f"""Ты — эксперт по сельскому хозяйству. Отвечай точно, кратко, на русском языке, используя только приведённые материалы.

Контекст:
{context_text}

Вопрос: {query}
Ответ:"""

    tokens_prompt = count_tokens(prompt)

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "Ты — агроном-консультант. Отвечай точно, на русском."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()
        tokens_completion = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        # Расчёт стоимости
        cost_usd = (
            (tokens_prompt / 1000) * current_model_prices["prompt"] +
            (tokens_completion / 1000) * current_model_prices["completion"]
        )

        log_query(query, answer, tokens_prompt, tokens_completion, cost_usd)

        # Добавляем статистику в ответ (опционально)
        answer += f"\n\nℹ️ Ответ сгенерирован на основе {len(context_chunks)} источников. Стоимость: ${cost_usd:.4f}"

        return answer, tokens_prompt, tokens_completion, cost_usd

    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        fallback = "Извините, не удалось сгенерировать ответ из-за временной ошибки."
        log_query(query, fallback, 0, 0, 0.0)
        return fallback, 0, 0, 0.0

# === Основная функция RAG ===
def get_answer(query: str) -> str:
    """
    Основной интерфейс для бота.
    Возвращает только строку — ответ.
    """
    # Проверяем кэш
    cached = get_cached_response(query)
    if cached:
        print("🔁 Используем кэш")
        return cached

    try:
        index, chunks = load_faiss_index()
        context_chunks = search_chunks(query, index, chunks, top_k=5)
        if not context_chunks:
            return "❌ Не найдено релевантных документов по вашему запросу."

        answer, _, _, _ = generate_answer(query, context_chunks)
        return answer

    except FileNotFoundError as e:
        return f"❌ Ошибка: {e}. Запустите create_index.py."
    except Exception as e:
        print(f"❌ Ошибка в get_answer: {e}")
        return "Извините, произошла ошибка при обработке запроса."

# === CLI-режим для тестирования ===
if __name__ == "__main__":
    init_db()
    print("💡 RAG-система готова. Введите 'выход', чтобы завершить.")

    while True:
        try:
            user_query = input("\n❓ Вопрос: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["выход", "exit", "quit"]:
                print("👋 До свидания!")
                break

            answer = get_answer(user_query)
            print(f"\n💬 Ответ: {answer}")

        except KeyboardInterrupt:
            print("\n👋 Завершаем работу...")
            break
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")