from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
import streamlit as st
from PIL import Image
import googlemaps
import io
import base64
import json

# --- SET API KEYS ---
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")


# --- GPT-4o-mini PROMPT ENGINEERING ---
def analyse_landmark(image_data):
    base64_image = encode_image(image_data)
    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a travel AI that identifies landmarks and provides travel plans. Respond in JSON format with 'landmark_name', 'latitude', 'longitude', and 'travel_plan'."},
        {"role": "user", "content": [
            {"type": "text", "text": "Identify this landmark, its location, and suggest a 3-day travel plan including must-visit places, local food, and cultural experiences."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ]},
    ],
    response_format="json")

    try:
        data = json.loads(response.choices[0].message.content)
        return data
    except json.JSONDecodeError:
        return {"error": "Failed to parse AI response"}


# --- FETCH NEARBY ATTRACTIONS FROM GOOGLE MAPS ---
def get_nearby_places(lat, lon, place_type="tourist_attraction"):
    """Fetch nearby attractions within 5km of the landmark."""
    try:
        places = gmaps.places_nearby(location=(lat, lon), radius=5000, type=place_type)
        return [place["name"] for place in places.get("results", [])]
    except Exception as e:
        return [f"Error fetching places: {e}"]

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Travel Planner", layout="centered")

st.title("🗺️ AI Travel Planner")
st.write("Upload an image of a landmark, and I'll plan a trip for you!")

uploaded_file = st.file_uploader("Upload Landmark Image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📍 Uploaded Landmark", use_column_width=True)

    # Convert image to binary
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    # Analyze Image with GPT-4o-mini
    with st.spinner("Analyzing landmark... 🔍"):
        result = analyse_landmark(image_data)

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader(f"📍 Identified Landmark: {result['landmark_name']}")
        st.write(result["travel_plan"])
        # Retrieve lat long from LLM
        lat, lon = result["latitude"], result["longitude"]

        # Fetch Nearby Attractions using Lat Long on Gmaps
        with st.spinner("Fetching nearby attractions... 🏛️"):
            attractions = get_nearby_places(lat, lon)

        st.subheader("🌟 Nearby Attractions")
        st.write("\n".join(f"📍 {place}" for place in attractions))





