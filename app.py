import streamlit as st
import pandas as pd
import plotly.express as px
from generate_embeddings import save_course_embeddings
from preprocessing import preprocess_data
from sentence_transformers import SentenceTransformer, util
import joblib
import os

# Function to load cleaned data
def load_cleaned_data():
    file_path = 'cleaned_data.pkl'
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        print(f"File {file_path} not found. Preprocessing data now.")
        return None

@st.cache_data
def load_precomputed_embeddings():
    embeddings = save_course_embeddings()
    return embeddings

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def compute_user_similarity(_user_input, _course_embeddings, _model):
    """Compute cosine similarity between user input and course embeddings."""
    user_embedding = _model.encode(_user_input, convert_to_tensor=True)
    return util.cos_sim(user_embedding, _course_embeddings)[0]

# Main function
def main():
    #  Set page configuration
    st.set_page_config(
        page_title="Udemy Courses Data Analysis",
        page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAA7UlEQVR4AWP4TwMwQIaOGvrl5d+tKZ9XeXx8feM3dQw9PuHrHOH3CzU/ABGQsaf8y8/P/8g39OWV3/PlQMYByQ2Rn3YVfYaYDiRv7/pJpqHX1v+AmAI34tm5XyscPgIFj3Z/JdPQK6u+QzVjind+HfZJ6ujRowUFBRUVFciCU6dOBQquWLGCTEOXLl3KAAbIgv7+/kCRhoYGMg0FOgfT0NDQUCobOmqop6cnxFDykxTEUCADInLv3j2ICDBhkJ/4GWAgLi4OmDzh3C9fvpBv6MuXL2VkZBhQwdWrV6mQTU+fPr0UDIDhMIzqqFFDAUfsdt6zPZyZAAAAAElFTkSuQmCC",  # Udemy logo
        layout="wide",
    )
    st.markdown(
        """
        <style>
            .stApp {
                background-color: whitesmoke;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }
            button {
                background-color: #A435F0 !important;
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                cursor: pointer !important;
            }
            button:hover {
            background-color: #822bc0 !important;
            }
            .css-18e3th9 {
                background-color: #A435F0 !important;
                color: black !important;
                font-size: 20px;
                text-align: center;
            }
            h1, h2, h3, h4, h5, h6, p {
                color: black !important;
                text-align: center;
            }
            .block-container {
                max-width: 1000px;
                margin: auto;
            }
            [data-testid="stSidebar"] {
                background-color: #A435F0;
            }
            [data-testid="stSidebar"] .css-18e3th9 {
                color: black;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.image("https://frontends.udemycdn.com/frontends-marketplace-experience/staticx/udemy/images/v7/logo-udemy.svg", width=150)
    st.title("Udemy Courses Data Analysis & AI-Powered Recommendation System")
    # Load data and model
    cleaned_data = load_cleaned_data()
    if cleaned_data is None:
        print("Preprocessing data...")
        cleaned_data = preprocess_data("Udemy Courses.xlsx")
        print("Saving cleaned data...")
        joblib.dump(cleaned_data, "cleaned_data.pkl")

    course_embeddings = load_precomputed_embeddings()
    model = load_model()

    if "reset_count" not in st.session_state:
        st.session_state["reset_count"] = 0

    input_key = f"user_input_widget_{st.session_state['reset_count']}"

    user_input = st.text_input(
        "Tell us about your interest or field of study (e.g., 'I study computer science'): ",
        value="",  # Always start with an empty value
        key=input_key  # Dynamic key!
    )

    if st.button("Reset"):
        st.session_state["reset_count"] += 1  # Increment the reset counter
        st.rerun()
    # Get Recommendations Button
    if st.button("Get Recommendations"):
        if user_input:
            # Clear cached data when input changes
            compute_user_similarity.clear()
            try:
                with st.spinner("Finding the best courses for you..."):
                    similarities = compute_user_similarity(user_input, course_embeddings, model)
                    top_k = similarities.topk(k=5)

                recommended_courses = cleaned_data.iloc[top_k.indices.tolist()][['Course Name', 'Category', 'Course Rating']]

                if not recommended_courses.empty:
                    st.subheader(f"Recommended Courses for '{user_input}'")
                    st.write(recommended_courses)

                    # Filter data for visualizations
                    filtered_data = cleaned_data[cleaned_data['Course Name'].isin(recommended_courses['Course Name'])]

                    # Visualization: Category Distribution
                    st.subheader("Category Distribution")
                    category_counts = filtered_data['Category'].value_counts()
                    fig2 = px.pie(
                        names=category_counts.index,
                        values=category_counts.values,
                        title="Category Distribution for Recommended Courses"
                    )
                    st.plotly_chart(fig2)

                    # Visualization: Course Duration vs. Ratings
                    st.subheader("Course Duration vs. Ratings for Recommended Courses")
                    if 'Course_Duration' in filtered_data.columns and 'Course Rating' in filtered_data.columns:
                        fig3 = px.scatter(
                            filtered_data,
                            x='Course_Duration',
                            y='Course Rating',
                            size='Course_Duration',
                            color='Category',
                            hover_data=['Course Name'],
                            title="Course Duration vs. Course Ratings"
                        )
                        st.plotly_chart(fig3)
                else:
                    st.write("No courses found matching your interest. Try another phrase!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide an input before clicking the button.")



if __name__ == "__main__":
    main()
