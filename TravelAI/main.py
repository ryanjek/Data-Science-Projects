import streamlit as st
import io
from PIL import Image
from gpt_utils import identify_landmark_gpt, generate_initial_itinerary, refine_itinerary
from maps_utils import get_nearby_places, google_vision, get_coordinates

st.set_page_config(page_title="Travel Buddy", layout="centered")

st.title("🗺️ Travel Buddy")
st.write("Upload an image of a landmark, and I'll plan a trip for you!")

# Ensure GPT is only called once per image upload
if "landmark_name" not in st.session_state:
    st.session_state.landmark_name = None
    st.session_state.latitude = None
    st.session_state.longitude = None
    # Initial itinerary from GPT
    st.session_state.initial_itinerary = None  
    # Itinerary refined with user-selected attractions
    st.session_state.refined_itinerary = None  
    st.session_state.selected_attraction_type = None
    st.session_state.nearby_places = []

# Use streamlit file uploader to take in image.
uploaded_file = st.file_uploader("Upload Landmark Image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📍 Uploaded Image", use_container_width=True)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    image_data = img_byte_arr.getvalue()

    # Identify Landmark if not already in session state
    if st.session_state.landmark_name is None:
        with st.spinner("Identifying landmark... 🔍"):
            # 1) Try Google Vision first
            best_landmark, lat, lon = google_vision(image_data)

            # 2) If Vision fails, fallback to GPT
            if not best_landmark:
                gpt_result = identify_landmark_gpt(image_data)
                best_landmark = gpt_result.get("landmark_name")
                lat = gpt_result.get("lat")
                lon = gpt_result.get("lon")
                
                # If GPT provided a name but lat/lon are missing or None, you can optionally geocode:
                if best_landmark and (not lat or not lon):
                    lat, lon = get_coordinates(best_landmark)

            # If still no landmark name, show error
            if not best_landmark:
                st.error("🚨 Could not identify a landmark from the image.")
            else:
                # Store in session
                st.session_state.landmark_name = best_landmark
                st.session_state.latitude = lat
                st.session_state.longitude = lon

                st.subheader(f"📍 Identified Landmark: {best_landmark}")
                # st.write(f"🌍 **Latitude:** {lat}, **Longitude:** {lon}")

                # 3) Generate the initial 3-day itinerary
                with st.spinner("Planning the best itinerary for you!"):
                    st.session_state.initial_itinerary = generate_initial_itinerary(best_landmark)

    # Display the initial itinerary (if available)
    if st.session_state.initial_itinerary:
        st.markdown("## ✨ **3-Day Itinerary** ✨")
        for day in st.session_state.initial_itinerary:
            with st.expander(f"📅 **Day {day['day']}**"):
                st.write(f"**Morning:** {day['morning']}")
                st.write(f"**Afternoon**: {day['afternoon']}")
                st.write(f"**Evening**: {day['evening']}")
                st.write(f"🍽️ **Recommended Food:** {day['recommended_food']}")

    # Let the user choose the type of nearby attractions
    attraction_types = {
        "Tourist Attractions": "tourist_attraction",
        "Restaurants & Cafes": "restaurant",
        "Shopping Malls": "shopping_mall",
        "Museums": "museum",
        "Parks & Nature": "park",
        "Nightlife & Bars": "bar",
        "Zoos & Aquariums": "zoo",
        "Historical Sites": "church",
    }

    # Only show this if  have coordinates
    if st.session_state.latitude and st.session_state.longitude:
        selected_type = st.selectbox(
            "🎯 Choose Nearby Attractions",
            list(attraction_types.keys()),
            index=0
        )

        # When the user picks a new type, fetch from Google Places
        if selected_type != st.session_state.selected_attraction_type:
            with st.spinner(f"Fetching {selected_type}... 🏛️"):
                attraction_names = get_nearby_places(
                    st.session_state.latitude,
                    st.session_state.longitude,
                    attraction_types[selected_type]
                )
                st.session_state.selected_attraction_type = selected_type
                st.session_state.nearby_places = attraction_names

        # Show Nearby Places
        st.subheader(f"🌟 **Nearby {selected_type}**")
        if st.session_state.nearby_places:
            # Let user pick up to 5
            selected_places = st.multiselect(
                "Select up to 5 attractions to include in your itinerary",
                st.session_state.nearby_places
            )
            if len(selected_places) > 5:
                st.warning("Please select no more than 5 attractions.")

            # Refine itinerary after user selects
            if st.button("Refine Itinerary"):
                if not selected_places:
                    st.warning("No attractions selected.")
                else:
                    with st.spinner("Enhancing itinerary with selected attractions... ✈️"):
                        st.session_state.refined_itinerary = refine_itinerary(
                            st.session_state.landmark_name,
                            st.session_state.initial_itinerary,
                            selected_places
                        )

        else:
            st.warning(f"No {selected_type} found nearby.")

    # Display Refined Itinerary
    if st.session_state.refined_itinerary:
        st.markdown("## ✨ **Personalised Itinerary** ✨")
        for day in st.session_state.refined_itinerary:
            with st.expander(f"📅 **Day {day['day']}**"):
                st.write(f"**Morning:** {day['morning']}")
                st.write(f"**Afternoon**: {day['afternoon']}")
                st.write(f"**Evening**: {day['evening']}")
                st.write(f"🍽️ **Recommended Food:** {day['recommended_food']}")
