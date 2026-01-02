import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. MUST DEFINE THIS AT THE TOP (Matches preprocesser.ipynb)
# ---------------------------------------------------------
def cuisine_tokenizer(text):
    return text.split(',')

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_data():
    try:
        # Load the cleaned dataframe
        df = pd.read_csv('cleaned_data.csv')
        # Load the encoders
        with open('encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return df, encoders
    except FileNotFoundError:
        return None, None

# --- Page Setup ---
st.set_page_config(page_title="Swiggy Recommender", layout="wide")
st.title("🍔 Swiggy Restaurant Recommendation System")

df, encoders = load_data()

if df is None:
    st.error("🚨 Data files not found. Please run your preprocessing script first!")
    st.stop()

# Extract encoders
city_encoder = encoders['city_encoder']
cuisine_vectorizer = encoders['cuisine_vectorizer']

# ---------------------------------------------------------
# 3. SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Customize Your Search")

# Search by Cleaned City
unique_cities = sorted(df['city'].unique())
selected_city = st.sidebar.selectbox("Select City", unique_cities)

# Search by Cuisine
available_cuisines = sorted(cuisine_vectorizer.get_feature_names_out())
selected_cuisines = st.sidebar.multiselect("Select Cuisines (Optional)", available_cuisines)

# Budget and Rating
max_budget = st.sidebar.slider(
    "Maximum Cost for Two (₹)", 
    min_value=int(df['cost'].min()), 
    max_value=2000, 
    value=1000,
    step=50
)

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5)

# ---------------------------------------------------------
# 4. RECOMMENDATION ENGINE
# ---------------------------------------------------------
if st.sidebar.button("Show Recommendations"):
    
    # Filter 1: Hard constraints (City, Budget, Rating)
    # This uses the 'city' column which we cleaned in preprocess.py
    mask = (df['city'] == selected_city) & (df['cost'] <= max_budget) & (df['rating'] >= min_rating)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.warning("No matches found! Try increasing your budget or lowering the rating.")
    else:
        # Filter 2: Content-Based Filtering (Cuisine Similarity)
        if selected_cuisines:
            # Vectorize user choice
            user_input_str = ",".join(selected_cuisines)
            user_vector = cuisine_vectorizer.transform([user_input_str])
            
            # Vectorize the cuisines of restaurants in the filtered list
            # We use the 'cuisine_str' column we created in preprocess
            filtered_df['cuisine_norm'] = filtered_df['cuisine'].apply(lambda x: str(x).replace(', ', ','))
            restaurant_vectors = cuisine_vectorizer.transform(filtered_df['cuisine_norm'])
            
            # Calculate Similarity
            scores = cosine_similarity(user_vector, restaurant_vectors).flatten()
            filtered_df['score'] = scores
            
            # Rank by Similarity first, then Rating
            results = filtered_df.sort_values(by=['score', 'rating'], ascending=False)
        else:
            # If no cuisine is selected, just show top rated
            results = filtered_df.sort_values(by='rating', ascending=False)

        # ---------------------------------------------------------
        # 5. DISPLAY RESULTS
        # ---------------------------------------------------------
        st.subheader(f"Top Matches in {selected_city}")
        
        for _, row in results.head(10).iterrows():
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Rating", f"{row['rating']} ⭐")
                    st.caption(f"Cost: ₹{int(row['cost'])}")
                with col2:
                    st.subheader(row['name'])
                    # SHOWING THE NEW 'AREA' COLUMN HERE
                    if row['area'] != row['city']:
                       location_text = f"📍 {row['area']}, {row['city']}"
                    else:
                       location_text = f"📍 {row['city']}"
                
                    st.write(f"**Location:** {location_text}")
                    st.write(f"**Cuisine:** {row['cuisine']}")
            # -----------------------
            
                    st.write(f"🏠 {row['address']}")
                    if row['link'] and str(row['link']) != 'nan':
                        st.markdown(f"[Order on Swiggy]({row['link']})")
                st.divider()
# Footer
with st.expander("Project Details"):
    st.write("Data Source: Swiggy Dataset")
    st.write(f"Total Restaurants: {len(df)}")
