# 🍔 Swiggy Restaurant Recommender

An end-to-end Machine Learning web application that provides personalized restaurant recommendations using Content-Based Filtering.

## 🚀 Key Technical Achievements
- Outlier Correction Identified and capped data entry errors (e.g., corrected price errors of ₹300,000+).
- Location Parsing Split Area, City strings to enable precise local recommendations.
- Cuisine Similarity Implemented a `CountVectorizer` with a custom tokenizer and Cosine Similarity to match user tastes.
- Clean Architecture Separated the data pipeline (preprocessing) from the user interface (Streamlit).

## 🛠️ Tech Stack
- Python (Pandas, NumPy)
- Scikit-Learn (Preprocessing & Similarity)
- Streamlit (Web Interface)
- Pickle (Model Serialization)

## 📂 Project Structure
- `preprocesser.ipynb` Cleans raw data and generates the similarity models.
- `app.py` The interactive recommendation dashboard.
- `cleaned_data.csv` The human-readable processed dataset.
-  encoded_data.csv is for the machine to calculate similarities
- `encoder.pkl` The saved ML objects for real-time processing.

## ⚙️ How to Run
1. Install dependencies `pip install pandas scikit-learn streamlit`
2. Run Preprocessing `python preprocesser.ipynb`
3. Start App `streamlit run app.py`
