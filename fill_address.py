import os
import time
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_ROLE_KEY)

# OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_address(name, latitude, longitude):
    prompt = (
        f"Ты туристический помощник. Назови примерный адрес для туристического объекта по координатам.\n"
        f"Название: {name}\n"
        f"Координаты: {latitude}, {longitude}\n"
        f"Ответь в формате: 'ул. такая-то, город такой-то' или просто 'район такой-то, город'."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помощник для путешествий."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Ошибка при адресе для '{name}': {e}")
        return None

def update_addresses():
    print("📍 Получаем объекты без адреса...")

    data = supabase.table("attractions") \
        .select("id, name, latitude, longitude, address") \
        .is_("address", "null") \
        .limit(1000) \
        .execute()

    attractions = data.data
    print(f"🔢 Найдено {len(attractions)} объектов\n")

    for item in attractions:
        name = item["name"]
        lat = item.get("latitude")
        lon = item.get("longitude")

        if not lat or not lon:
            print(f"⚠️ Пропущено '{name}' — нет координат")
            continue

        address = generate_address(name, lat, lon)
        if address:
            supabase.table("attractions") \
                .update({"address": address}) \
                .eq("id", item["id"]) \
                .execute()
            print(f"✅ {name} — {address}")
        else:
            print(f"⛔ {name} — ошибка генерации")

        time.sleep(1)

if __name__ == "__main__":
    update_addresses()
