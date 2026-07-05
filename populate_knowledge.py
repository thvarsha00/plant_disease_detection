import psycopg2
import requests
from remedies import REMEDIES

def get_embedding(text):
    try:
        r = requests.post("http://localhost:11434/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": text
        }, timeout=15)
        if r.status_code == 200:
            embedding = r.json().get("embedding")
            if embedding:
                return "[" + ",".join(map(str, embedding)) + "]"
    except Exception as e:
        pass
    return None

def populate():
    # Connect to the PostgreSQL database
    try:
        conn = psycopg2.connect(
            dbname="ollama_chat",
            user="postgres",
            password="postgres123",
            host="localhost",
            port="5432"
        )
        cursor = conn.cursor()
        print("Connected to database successfully.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # 1. Custom Farming Guides to insert
    guides = [
        {
            "title": "Mango Fruit Red Dots Cause & Cure",
            "content": (
                "Farming Guide: Mango Fruit Red Dots. Red dots or spots on mango fruits are commonly caused by Anthracnose "
                "(a fungal infection caused by Colletotrichum gloeosporioides) or bacterial black spot. Symptoms start as "
                "small, water-soaked, circular, or irregular lesions on leaves and fruit, which later turn brown, black, or red dots. "
                "Organic cure: Spray copper fungicide, liquid seaweed, or potassium bicarbonate. Prune branches to improve air "
                "circulation. Chemical cure: Apply chlorothalonil or mancozeb. Prevention: Apply neem oil before fruit set, rake and "
                "destroy fallen leaves."
            )
        },
        {
            "title": "Growing Perfect Bananas Guide",
            "content": (
                "Farming Guide: How to Grow Perfect Bananas. Bananas are heavy feeders requiring warm weather, rich, well-draining soil, "
                "constant moisture, and shelter from wind. 1. Soil: Ideal pH is 5.5 - 6.5. Mix organic compost or manure heavily into the "
                "planting hole. 2. Watering: Water deeply and keep soil consistently moist but never soggy to prevent root rot. 3. Nutrients: "
                "High potassium fertilizers (NPK 8-10-8 or similar) are essential for healthy banana fruiting. 4. Pruning: Keep only one main "
                "fruiting stem and one sucker (follower) per plant to maximize yield."
            )
        },
        {
            "title": "Precautions Before Crop Raising",
            "content": (
                "Farming Guide: Precautions Before Crop Raising. 1. Soil Testing: Always test the soil pH and nutrient levels beforehand "
                "to prepare target amendments. 2. Seed Selection: Choose certified, disease-resistant seeds/varieties. 3. Tillage: Till the "
                "field properly to control weeds, expose pests, and improve soil aeration. 4. Soil Treatment: Solarize the soil or mix in "
                "beneficial microbes like Trichoderma (a bio-fungicide) to prevent root rot and soil-borne wilt diseases. 5. Drainage: Set up "
                "proper drainage channels to avoid waterlogging."
            )
        }
    ]

    # Add remedies from remedies.py
    for key, data in REMEDIES.items():
        title = key.replace("___", " - ").replace("_", " ")
        summary = data.get("summary", "")
        organic = ", ".join(data.get("organic", []))
        chemical = ", ".join(data.get("chemical", []))
        prevention = ", ".join(data.get("prevention", []))

        content = f"KrishiSetu Verified Manual: For {title}, this is a disease with the following details. Summary: {summary} "
        if organic:
            content += f"Organic Treatments: {organic}. "
        if chemical:
            content += f"Chemical Treatments: {chemical}. "
        if prevention:
            content += f"Prevention: {prevention}."

        guides.append({"title": f"Treatment Guide - {title}", "content": content})

    # Target Website connection ID 3 (KrishiSetu AI)
    website_connection_id = 3

    # Clean old metadata for website_connection_id = 3 to avoid duplicates or mismatch
    cursor.execute("DELETE FROM website_metadata WHERE website_connection_id = %s", (website_connection_id,))
    
    print("Generating embeddings and inserting knowledge chunks...")
    inserted = 0
    for idx, guide in enumerate(guides):
        content = guide["content"]
        title = guide["title"]
        
        # Try to generate embedding
        embedding = get_embedding(content)
        if not embedding:
            print(f"Skipping embedding for chunk {idx+1}/{len(guides)}: nomic-embed-text not running or failed.")
            continue
            
        # Insert into website_metadata (for the widget chat RAG)
        cursor.execute(
            """
            INSERT INTO website_metadata (content, content_type, created_at, embedding, title, url, website_connection_id)
            VALUES (%s, %s, now(), %s, %s, %s, %s)
            """,
            (content, "GENERAL", embedding, title, "http://localhost:8501", website_connection_id)
        )
        
        # Also ensure it exists in general knowledge_chunk
        cursor.execute("SELECT id FROM knowledge_chunk WHERE content = %s", (content,))
        row = cursor.fetchone()
        if not row:
            cursor.execute("INSERT INTO knowledge_chunk (content, embedding) VALUES (%s, %s)", (content, embedding))
            
        inserted += 1

    conn.commit()
    print(f"Successfully processed and seeded {inserted} knowledge chunks into website_metadata (connection ID {website_connection_id}) and knowledge_chunk table with vector embeddings.")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    populate()
