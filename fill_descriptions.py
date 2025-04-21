import os
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ⛓️ Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ROLE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_ROLE_KEY)

# 🤖 OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-3.5-turbo"

def generate_description(name, category, city):
    prompt = (
        f"Напиши короткое и красивое описание для туристического объекта.\n"
        f"Название: {name}\n"
        f"Категория: {category}\n"
        f"Город: {city}\n\n"
        f"Описание:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Ты помощник для путешествий."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка GPT при описании '{name}': {e}")
        return None

def update_descriptions():
    print("🔍 Получаем аттракционы без описания...")

    data = supabase.table("attractions") \
        .select("id, name, category, city, description") \
        .or_("description.is.null,description.eq.Добавлено из Яндекс API") \
        .limit(1000) \
        .execute()

    attractions = data.data
    print(f"🧠 Нужно сгенерировать описания для {len(attractions)} объектов\n")

    for item in attractions:
        name = item["name"]
        category = item.get("category") or "достопримечательность"
        city = item.get("city") or "город"
        new_description = generate_description(name, category, city)

        if new_description:
            supabase.table("attractions") \
                .update({"description": new_description}) \
                .eq("id", item["id"]) \
                .execute()
            print(f"✅ {name} — описание обновлено")
        else:
            print(f"⛔ {name} — ошибка генерации")

if __name__ == "__main__":
    update_descriptions()