# main.py
from fastapi import FastAPI, Request
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI  # 🔁 Новый импорт OpenAI (из langchain_community)
from openai import OpenAI as OpenAIClient     # 🤖 Для embedding
from supabase import create_client
import os
from dotenv import load_dotenv
from langchain.tools import tool
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

# 🔐 Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🔐 OpenAI
client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))  # для embedding
llm = OpenAI(temperature=0)  # для LangChain GPT

# 🎯 Тул: поиск по категориям
def search_attractions(query: str) -> str:
    try:
        city, category = query.split('|')
    except ValueError:
        return "Формат запроса должен быть: город|категория"

    response = supabase.table('attractions')\
        .select("name, description, category, city, rating")\
        .ilike("city", f"%{city.strip()}%")\
        .ilike("category", f"%{category.strip()}%")\
        .order("rating", desc=True)\
        .limit(5)\
        .execute()

    if not response.data:
        return "Ничего не найдено по запросу"

    return "\n\n".join([
        f"{r['name']} ({r['category']}, {r['city']}) – {r['description'] or 'без описания'}"
        for r in response.data
    ])

# 🎯 Тул: векторный поиск
def semantic_search(query: str) -> str:
    try:
        embedding_response = client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        embedding = embedding_response.data[0].embedding

        response = supabase.rpc("match_attractions", {
            "query_embedding": embedding,
            "match_threshold": 0.85,
            "match_count": 5
        }).execute()

        if not response.data:
            return "Ничего не найдено по смысловому запросу"

        return "\n\n".join([
            f"{r['name']} ({r['category']}, {r['city']}) – {r['description'] or 'без описания'}"
            for r in response.data
        ])
    except Exception as e:
        print(f"Ошибка semantic_search: {e}")
        return "Произошла ошибка при поиске"
    
def geocode_address(location_text: str) -> tuple[float, float]:
    prompt = (
        f"Ты помощник путешественника. Пользователь указал своё местоположение: {location_text}.\n"
        f"Предположи примерные координаты в формате 'LAT, LON' — только цифры, никаких слов. Например: '43.5855, 39.7231'."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты отвечаешь всегда только координатами."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=30
        )
        coords = response.choices[0].message.content.strip().split(",")
        return float(coords[0]), float(coords[1])
    except Exception as e:
        print(f"Ошибка geocode_address: {e}")
        return None, None

def find_nearby_attractions(lat: float, lon: float, radius_km=3.0) -> list:
    try:
        response = supabase.rpc("find_nearby_attractions", {
            "origin_lat": lat,
            "origin_lon": lon,
            "radius_km": radius_km
        }).execute()

        return response.data or []
    except Exception as e:
        print(f"Ошибка поиска точек рядом: {e}")
        return []
    
def generate_route_description(points: list) -> str:
    if not points:
        return "Ничего не найдено рядом для маршрута."

    walking = [p for p in points if p['distance'] <= 2]
    driving = [p for p in points if p['distance'] > 2]

    def format_block(title: str, list_: list) -> str:
        if not list_:
            return f"{title}: ничего не найдено."
        return f"{title}:\n" + "\n".join([
            f"- {p['name']} ({p['category']}, {round(p['distance'], 2)} км, рейтинг {p.get('rating') or '—'})\n  {p.get('address', 'адрес неизвестен')}.\n  {p['description'] or ''}"
            for p in list_
        ])

    walking_text = format_block("Пешие маршруты", walking)
    driving_text = format_block("Места для поездки на машине", driving)

    prompt = (
        f"Ты помощник для построения маршрута с детьми. Пользователь находится в точке A.\n\n"
        f"{walking_text}\n\n"
        f"{driving_text}\n\n"
        f"Составь осмысленный маршрут: сначала предложи пешие варианты, потом — куда можно поехать.\n"
        f"Напиши легко, человечно, с понятной структурой. Максимум 2 абзаца."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка генерации описания маршрута: {e}")
        return "Ошибка генерации маршрута."
    
def plan_walk_from_location(location_text: str) -> str:
    lat, lon = geocode_address(location_text)
    if not lat or not lon:
        return "Не удалось определить координаты"

    nearby = find_nearby_attractions(lat, lon)
    return generate_route_description(nearby)

def find_child_friendly_places(lat: float, lon: float) -> str:
    points = find_nearby_attractions(lat, lon)
    child_related = [p for p in points if 'дет' in (p['description'] or '').lower() or 'парк' in (p['category'] or '').lower()]
    return "\n".join([f"{p['name']} – {p['description']}" for p in child_related])

def semantic_search_nearby(query: str, lat: float, lon: float, radius_km: float = 2.0) -> str:
    try:
        # Получаем embedding
        embedding_response = client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        embedding = embedding_response.data[0].embedding

        # Векторный поиск в Supabase (через RPC)
        response = supabase.rpc("match_attractions", {
            "query_embedding": embedding,
            "match_threshold": 0.82,
            "match_count": 8,
            "user_lat": lat,
            "user_lon": lon,
            "radius_km": radius_km
        }).execute()

        if not response.data:
            return "Ничего не найдено поблизости."

        return "\n\n".join([
            f"{r['name']} ({r['category']}, {r['city']}) — {round(r['distance_km'], 2)} км — {r['description'] or 'без описания'}"
            for r in response.data
        ])

    except Exception as e:
        print(f"Ошибка semantic_search_nearby: {e}")
        return "Произошла ошибка при поиске."

def semantic_walk_search(query: str) -> str:
    # Пока координаты захардкожены (Морской вокзал Сочи)
    lat = 43.5814
    lon = 39.7181
    return semantic_search_nearby(query, lat, lon, radius_km=2)

def semantic_drive_search(query: str) -> str:
    lat = 43.5814
    lon = 39.7181
    return semantic_search_nearby(query, lat, lon, radius_km=50)

def semantic_geo_search(query: str) -> str:
    try:
        user_lat = 43.5814  # пока захардкожено — Морской вокзал
        user_lon = 39.7181
        radius_km = 3.0

        # 1. Получаем embedding
        embedding_response = client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        embedding = embedding_response.data[0].embedding

        # 2. RPC-вызов
        response = supabase.rpc("match_attractions", {
            "query_embedding": embedding,
            "match_threshold": 0.82,
            "match_count": 8,
            "user_lat": user_lat,
            "user_lon": user_lon,
            "radius_km": radius_km
        }).execute()

        data = response.data
        if not data:
            return "Ничего не найдено рядом с вами."

        # 3. Форматируем результат красиво
        result = "Вот несколько подходящих мест поблизости:\n\n"
        for p in data:
            result += (
                f"📍 *{p['name']}* ({p['category']}, {round(p['distance_km'], 2)} км)\n"
                f"⭐️ Рейтинг: {p.get('rating', '—')}  \n"
                f"📬 Адрес: {p.get('address', 'не указан')}  \n"
                f"📝 {p['description'] or 'Нет описания'}\n\n"
            )

        return result.strip()

    except Exception as e:
        print(f"Ошибка semantic_geo_search: {e}")
        return "Произошла ошибка при геопоиске"

# 🧠 Агрегируем Tool'ы
tools = [
    Tool(
        name="search_attractions",
        func=search_attractions,
        description=(
            "Use to find attractions by city and category. "
            "Format: 'Сочи|парк'. Categories: парк, природа, еда, достопримечательность, искусство, активный отдых"
        )
    ),
    Tool(
        name="semantic_search",
        func=semantic_search,
        description="Use to find attractions by meaning. Natural query like 'куда сходить с детьми в Сочи'"
    ),
    Tool(
        name="plan_walk_from_location",
        func=plan_walk_from_location,
        description="Планирует короткий маршрут от указанного места (например, 'Я в отеле Рэдисон в Сочи')",
        return_direct=True
    ),
    Tool(  # ты дважды вставил search_attractions — этот дубликат удаляю
        name="semantic_walk_search",
        func=semantic_walk_search,
        description="Ищи интересные места в пешей доступности (до 2 км). Отвечай так, как есть в выводе функции — не сокращай названия, не выкидывай описание.",
        return_direct=True
    ),
    Tool(
        name="semantic_drive_search",
        func=semantic_drive_search,
        description="Найти места для поездки на машине (до 50 км), например: 'куда можно съездить с детьми на целый день'",
        return_direct=True
    ),
    Tool(
        name="semantic_geo_search",
        func=semantic_geo_search,
        description="Найди подходящие места рядом с точкой. Используй для смыслового поиска с гео-фильтрацией.",
        return_direct=True
    )
]

# 🤖 GPT агент
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={
        "system_message": (
            "Ты тревел-бот, который помогает людям планировать прогулки, отдых и досуг в городе.\n"
            "Если один из инструментов (Tool) вернул список мест, их описание, адреса, рейтинги и т.д. — "
            "**не переписывай** этот список своими словами.\n"
            "Просто **вставь ответ инструмента в результат как есть**, сохраняя формат, эмодзи, расстояния и порядок.\n"
            "Не сокращай и не заменяй текст. Это важно для точной передачи информации.\n"
            "Если инструмент вернул markdown или форматированный текст — оставь его в таком же виде."
        )
    }
)

# 🚀 FastAPI endpoint
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ❗ Или укажи фронт: ["https://injoy-ten.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

@app.post("/chat", response_class=PlainTextResponse)
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    result = agent.run(query)
    return result