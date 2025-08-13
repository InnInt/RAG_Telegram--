import os

# === Путь к папке с исходными документами ===
KNOWLEDGE_PATH = os.path.join("data", "knowledge")  # было: "agrobot", "data", "knowledge"

# === Папка для сохранения FAISS-индекса ===
FAISS_INDEX_PATH = os.path.join("embeddings", "faiss_index")  # можно оставить

# === Файл базы SQLite для логов ===
DB_PATH = os.path.join("data", "logs.db")  # было: "agrobot", "data", "logs.db"

# === Параметры OpenAI ===
OPENAI_API_KEY = "sk-proj-IDQPTpu9KL_p4j_iltOgaP-8ITyZVKbnLfaaHyJt11O-fKCKuTvVwRVBmD6U6hadCf1JDl89myT3BlbkFJfltNhQ-NJRdNCX6LdTPtysjAncnJKeOI_752Gh3sNX2gpuuM-ksFd2eSkSePqEvPxp4W9vc6oAAA"  # вставь свой ключ
EMBEDDING_MODEL = "text-embedding-3-small"  # или text-embedding-3-large
GPT_MODEL = "gpt-4o-mini"  # <-- сначала объявляем

CHAT_MODEL = GPT_MODEL     # <-- теперь можно использовать

# === Telegram ===
TELEGRAM_BOT_TOKEN = "7563364734:AAHUTF7dfPmfLY_xR7qVVGfKS0GdXLo1n-MAA"
BOT_TOKEN = TELEGRAM_BOT_TOKEN

# === Чанкование текста ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Цены моделей (пример для оценки стоимости) ===
MODEL_PRICES = {
    "gpt-4o-mini": {
        "input": 0.00015 / 1000,   # $ за 1K токенов
        "output": 0.00060 / 1000
    },
    "gpt-4": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000
    }
}
