import os
import time
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.getenv("SUPABASE_KEY")
client = create_client(SUPABASE_URL, SUPABASE_ROLE_KEY)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str) -> list[float]:
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Ошибка генерации embedding: {e}")
        return None

def generate_embeddings():
    print("📡 Получаем аттракционы без embedding...")

    result = client.table("attractions").select("id, description, embedding, city").execute()

    rows = [
        r for r in result.data
        if r["embedding"] is None
        and r["description"]
        and "Яндекс" not in r["description"]
    ]

    print(f"🔢 Найдено {len(rows)} записей для обработки")

    for row in rows:
        desc = row["description"]
        embedding = get_embedding(desc)
        if embedding:
            client.table("attractions") \
                .update({"embedding": embedding}) \
                .eq("id", row["id"]) \
                .execute()
            print(f"✅ Обновлено: {row['id']}")
        else:
            print(f"⛔ Пропущено: {row['id']}")
        time.sleep(1)

if __name__ == "__main__":
    generate_embeddings()
