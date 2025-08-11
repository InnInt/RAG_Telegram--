import os
import pickle
import hashlib
import sqlite3
import faiss
import docx
import fitz  # PyMuPDF
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from config import KNOWLEDGE_PATH, FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY, EMBEDDING_MODEL, DB_PATH

# Инициализация OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# ==== Чтение разных форматов ====
def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ==== Хэш содержимого ====
def file_hash(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()

# ==== SQLite для отслеживания файлов ====
def init_db():
    # Создаём папку data/, если её нет
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    print(f"🔧 Гарантируем существование папки: {db_dir}")

    # Подключаемся к базе
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"✅ Подключено к БД: {DB_PATH}")
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")
        raise

    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_path TEXT PRIMARY KEY,
            modified_time TEXT,
            hash TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_indexed_files():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT file_path, modified_time, hash FROM indexed_files")
    data = {row[0]: {"modified_time": row[1], "hash": row[2]} for row in cur.fetchall()}
    conn.close()
    return data

def update_indexed_file(file_path, modified_time, file_hash_value):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO indexed_files (file_path, modified_time, hash)
        VALUES (?, ?, ?)
    """, (file_path, modified_time, file_hash_value))
    conn.commit()
    conn.close()

# ==== Чанкование текста ====
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)

# ==== Эмбеддинги ====
def embed_texts(chunks):
    vectors = []
    for chunk in chunks:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk
        )
        vectors.append(resp.data[0].embedding)
    return np.array(vectors).astype("float32")

# ==== Основная логика ====
def main():
    print("🚀 Запуск индексации...")
    init_db()
    indexed_files = get_indexed_files()

    # Загружаем или создаём FAISS индекс
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        index = faiss.read_index(os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        with open(os.path.join(FAISS_INDEX_PATH, "index.pkl"), "rb") as f:
            chunks_all = pickle.load(f)
    else:
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        index = None
        chunks_all = []

    new_chunks = []
    new_vectors = []

    for root, _, files in os.walk(KNOWLEDGE_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.lower().split(".")[-1]

            if ext not in ["docx", "pdf", "txt"]:
                continue

            modified_time = str(os.path.getmtime(file_path))

            # Чтение файла
            if ext == "docx":
                content = read_docx(file_path)
            elif ext == "pdf":
                content = read_pdf(file_path)
            elif ext == "txt":
                content = read_txt(file_path)

            content_hash = file_hash(content)

            # Проверка: новый или изменённый
            if file_path in indexed_files:
                if indexed_files[file_path]["hash"] == content_hash:
                    continue  # Файл не изменился

            print(f"📄 Индексация: {file}")

            chunks = create_chunks(content)
            vectors = embed_texts(chunks)

            new_chunks.extend(chunks)
            new_vectors.extend(vectors)

            update_indexed_file(file_path, modified_time, content_hash)

    if new_chunks:
        print(f"➕ Добавляем {len(new_chunks)} чанков в индекс...")

        new_vectors_np = np.array(new_vectors).astype("float32")

        if index is None:
            index = faiss.IndexFlatL2(len(new_vectors_np[0]))

        index.add(new_vectors_np)
        chunks_all.extend(new_chunks)

        faiss.write_index(index, os.path.join(FAISS_INDEX_PATH, "index.faiss"))
        with open(os.path.join(FAISS_INDEX_PATH, "index.pkl"), "wb") as f:
            pickle.dump(chunks_all, f)

        print("✅ Индекс обновлён")
    else:
        print("ℹ️ Новых или изменённых файлов нет")

if __name__ == "__main__":
    main()
