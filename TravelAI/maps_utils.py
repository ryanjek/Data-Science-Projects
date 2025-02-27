import googlemaps
from config import GOOGLE_MAPS_API_KEY, GOOGLE_APPLICATION_CREDENTIALS
from google.cloud import vision
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# Geocode the coordinate from the landmark indentified by gpt4o-mini
def get_coordinates(place_name):
    try:
        # Exception catching to seee if there is results from geocoding
        geocode_result = gmaps.geocode(place_name)

        if geocode_result and "geometry" in geocode_result[0]:
            location = geocode_result[0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            # When there is no lat and Long
            return None, None
    except Exception as e:
        return None, None

# Cannot use keyword=name here as it gives back the exact landmark.
def get_nearby_places( lat, lon, attraction):
    try:
        response = gmaps.places_nearby(
            location=(lat, lon), 
            radius=5000, 
            type=attraction, 
        )

        if "results" in response and response["results"]:
            place_names = [place.get("name", "Unknown Place") for place in response["results"]]
            return place_names 
        else:
            return ["No places found."]
    except Exception as e:
        return [f"Error fetching places: {e}"]
    
# Use google vision to identify landmarks
def google_vision(image_data):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations

    if not landmarks:
        # No landmark detected
        return None, None, None

    # Take the first result only.
    top_landmark = landmarks[0]
    name = top_landmark.description
    lat = top_landmark.locations[0].lat_lng.latitude
    lon = top_landmark.locations[0].lat_lng.longitude

    # Return top landmark’s info
    return name, lat, lon





# --- ENABLE GOOGLE APIs ---
# Places API (For fetching nearby places)
# Geocoding API (For getting latitude/longitude from a place name)
# Maps JavaScript API (If using Google Maps on a website)
# Directions API (If getting route directions)