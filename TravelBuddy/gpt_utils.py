from openai import OpenAI
import base64
import json
from config import OPENAI_API_KEY


# Convert  image into a Base64 string to be used as inputs
def encode_image(image_data):
    return base64.b64encode(image_data).decode("utf-8")

# --- Landmark identifier ---
client = OpenAI(api_key=OPENAI_API_KEY)    
def identify_landmark_gpt(image_data):
    """
    1) Sends the uploaded image to GPT-4o-mini.
    2) GPT attempts to identify the landmark name and provide approximate latitude & longitude.
    3) Returns a dict with "landmark_name", "lat", "lon".
    """
    base64_image = encode_image(image_data)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a travel-savvy AI. You receive an image in base64 and must identify the landmark. "
                    "Also provide approximate latitude and longitude if you can infer them. Return valid JSON:\n"
                    "{\n"
                    "  \"landmark_name\": \"string\",\n"
                    "  \"lat\": \"number or string\",\n"
                    "  \"lon\": \"number or string\"\n"
                    "}\n"
                    "No extra keys, no markdown. If uncertain, put null or empty strings for lat/lon."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the landmark in this image and provide approximate latitude & longitude."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                ],
            },
        ],
    )

    try:
        raw_content = response.choices[0].message.content.strip()
        data = json.loads(raw_content)
        return data
    except (json.JSONDecodeError, IndexError):
        return {
            "landmark_name": None,
            "lat": None,
            "lon": None
        }


def generate_initial_itinerary(landmark_name):
    """
    Generates a 3-day itinerary for the identified landmark.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced AI travel assistant. Your task is to generate a detailed 3-day travel itinerary "
                    "for a given landmark. Your response must be in valid JSON format strictly following this structure:\n"
                    "{\n"
                    "  \"landmark_name\": \"string\",\n"
                    "  \"travel_plan\": [\n"
                    "    {\n"
                    "      \"day\": 1,\n"
                    "      \"morning\": \"string\",\n"
                    "      \"afternoon\": \"string\",\n"
                    "      \"evening\": \"string\",\n"
                    "      \"recommended_food\": \"string\"\n"
                    "    },\n"
                    "    ... (repeat for day 2 and 3)\n"
                    "  ]\n"
                    "}\n"
                    "Do not include extra text, markdown, explanations, or formatting outside of valid JSON."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generate a 3-day itinerary for visiting {landmark_name}, including must-see attractions, local food, and activities.",
                    }
                ],
            },
        ],
    )

    try:
        response_content = response.choices[0].message.content.strip()
        data = json.loads(response_content)

        if "travel_plan" in data and isinstance(data["travel_plan"], list):
            return data["travel_plan"]
        else:
            return [{"day": 1, "morning": "Error in GPT response", "afternoon": "Error", "evening": "Error", "recommended_food": "Error"}]

    except json.JSONDecodeError:
        return [{"day": 1, "morning": "Error parsing GPT response", "afternoon": "Error", "evening": "Error", "recommended_food": "Error"}]


def refine_itinerary(landmark_name, base_itinerary, user_selected):
    """
    Refines the itinerary by incorporating user-selected attractions.
    """
    prompt = (
        f"You are an expert travel planner. The user has a 3-day itinerary for visiting {landmark_name}:\n\n"
        f"{json.dumps(base_itinerary, indent=2)}\n\n"
        f"The user wants to include these additional attractions: {', '.join(user_selected)}.\n"
        "Modify the itinerary to incorporate these attractions while maintaining a balanced experience. "
        "Ensure the itinerary remains structured as a JSON list using the exact format:\n"
        "{\n"
        "  \"travel_plan\": [\n"
        "    {\n"
        "      \"day\": 1,\n"
        "      \"morning\": \"string\",\n"
        "      \"afternoon\": \"string\",\n"
        "      \"evening\": \"string\",\n"
        "      \"recommended_food\": \"string\"\n"
        "    },\n"
        "    ... (repeat for day 2 and 3)\n"
        "  ]\n"
        "}\n"
        "Do not include explanations, markdown, or text outside of this strict JSON format."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_content = response.choices[0].message.content.strip()
        data = json.loads(response_content)

        if "travel_plan" in data and isinstance(data["travel_plan"], list):
            return data["travel_plan"]
        else:
            return base_itinerary  # Fallback to the original itinerary

    except json.JSONDecodeError:
        return base_itinerary  # Return original if GPT fails









# def analyse_landmark_gpt(image_data):
#     base64_image = encode_image(image_data)

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are an experienced AI travel assistant. Identify the landmark in an image and return a JSON response "
#                     "strictly in this format:\n"
#                     "{\n"
#                     "  \"landmark_name\": \"string\",\n"
#                     "  \"travel_plan\": \"string\"\n"
#                     "}\n"
#                     "Do not add extra keys, markdown, or explanations. Only output valid JSON."
#                 ),
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Identify the landmark and provide a detailed 3-day itinerary including local food, must see attractions and activities.",
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                     },
#                 ],
#             },
#         ],
#     )

#     # Ensure valid JSON parsing
#     try:
#         response_content = response.choices[0].message.content.strip()
#         data = json.loads(response_content)
#         return data
#     except json.JSONDecodeError:
#         return {"error": "Failed to parse AI response, likely due to invalid JSON format."}