# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from preprocessing import preprocess_data  # Import the preprocessing function
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer, util

# # Load the sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient and lightweight model

# # Set up the Streamlit app
# def main():
#     # Set page configuration
#     st.set_page_config(
#         page_title="Udemy Courses Data Analysis",
#         page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAA7UlEQVR4AWP4TwMwQIaOGvrl5d+tKZ9XeXx8feM3dQw9PuHrHOH3CzU/ABGQsaf8y8/P/8g39OWV3/PlQMYByQ2Rn3YVfYaYDiRv7/pJpqHX1v+AmAI34tm5XyscPgIFj3Z/JdPQK6u+QzVjind+HfZJ6ujRowUFBRUVFciCU6dOBQquWLGCTEOXLl3KAAbIgv7+/kCRhoYGMg0FOgfT0NDQUCobOmqop6cnxFDykxTEUCADInLv3j2ICDBhkJ/4GWAgLi4OmDzh3C9fvpBv6MuXL2VkZBhQwdWrV6mQTU+fPr0UDIDhMIzqqFFDAUfsdt6zPZyZAAAAAElFTkSuQmCC",  # Udemy logo
#         layout="wide",
#     )

#     # Apply custom CSS for background, header colors, and centering elements
#     st.markdown(
#         """
#         <style>
#             .stApp {
#                 background-color: whitesmoke;
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 flex-direction: column;
#             }
#             .css-18e3th9 {
#                 background-color: #A435F0 !important;
#                 color: black !important;
#                 font-size: 20px;
#                 text-align: center;
#             }
#             h1, h2, h3, h4, h5, h6, p {
#                 color: black !important;
#                 text-align: center;
#             }
#             .block-container {
#                 max-width: 1000px;
#                 margin: auto;
#             }
#             [data-testid="stSidebar"] {
#                 background-color: #A435F0;
#             }
#             [data-testid="stSidebar"] .css-18e3th9 {
#                 color: black;  /* Adjust text color if necessary */
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.image("https://frontends.udemycdn.com/frontends-marketplace-experience/staticx/udemy/images/v7/logo-udemy.svg", width=150)
#     st.title("Udemy Courses Data Analysis and AI-Powered Recommendation System")

#     # Load the raw data
#     file_path = "Udemy Courses.xlsx"  # Replace with the actual file name
#     with st.spinner("Loading the raw data..."):
#         raw_data = pd.read_excel(file_path)

#     # Display the raw data
#     # st.subheader("Uncleaned Data")
#     # st.write(raw_data)

#     # Preprocess the data
#     with st.spinner("Processing the data..."):
#         cleaned_data = preprocess_data(file_path)

#     # Display the cleaned data
#     # st.subheader("Cleaned Data")
#     # st.write(cleaned_data)

#     # User input for course recommendation
#     st.header("Course Recommendation System")
#     user_input = st.text_input("Tell us about your interest or field of study (e.g., 'I study computer science'):")

#     if user_input:
#         # Compute embeddings for user input and course names
#         user_embedding = model.encode(user_input, convert_to_tensor=True)
#         course_embeddings = model.encode(cleaned_data['Course Name'].fillna('').tolist(), convert_to_tensor=True)

#         # Compute cosine similarity
#         similarities = util.cos_sim(user_embedding, course_embeddings)[0]
#         top_k = similarities.topk(k=5)  # Get top 5 matches

#         # Extract top matching courses
#         recommended_courses = cleaned_data.iloc[top_k.indices.tolist()][['Course Name', 'Category', 'Course Rating']]
        
#         if not recommended_courses.empty:
#             st.subheader(f"Recommended Courses for '{user_input}'")
#             st.write(recommended_courses)
#         else:
#             st.write("No courses found matching your interest. Try another phrase!")

#     # Data Visualization
#     st.header("Data Visualization")

#     # Visualization: Bar Chart of Course Ratings
#     if 'Rating' in cleaned_data.columns:
#         st.subheader("Course Ratings Distribution")
#         fig1 = px.histogram(cleaned_data, x='Rating', nbins=20, title="Distribution of Course Ratings")
#         st.plotly_chart(fig1)

#     # Visualization: Pie Chart of Categories
#     if 'Category' in cleaned_data.columns:
#         st.subheader("Category Distribution")
#         category_counts = cleaned_data['Category'].value_counts()
#         fig2 = px.pie(
#             names=category_counts.index,
#             values=category_counts.values,
#             title="Distribution of Course Categories"
#         )
#         st.plotly_chart(fig2)

#     # Visualization: Scatter Plot of Price vs. Ratings
#     if 'Price' in cleaned_data.columns and 'Rating' in cleaned_data.columns:
#         st.subheader("Price vs. Ratings")
#         fig3 = px.scatter(
#             cleaned_data,
#             x='Price',
#             y='Rating',
#             size='Price',
#             color='Category' if 'Category' in cleaned_data.columns else None,
#             hover_data=['Course Title'] if 'Course Title' in cleaned_data.columns else None,
#             title="Price vs. Ratings"
#         )
#         st.plotly_chart(fig3)

#     # Top 5 courses with highest ratings
#     st.subheader("Top 5 Courses with Highest Ratings by Category")

#     # Add selectbox for category selection
#     category_top_courses = st.selectbox("Select a Category for Top Courses", cleaned_data['Category'].unique(), key="top_courses")

#     # Filter data by selected category
#     filtered_category_df = cleaned_data[cleaned_data['Category'] == category_top_courses]

#     # Get the top 5 courses in the selected category by rating
#     top_5_courses_by_category = filtered_category_df.nlargest(5, 'Course Rating')[['Course Name', 'Course Rating']]

#     # Display the top 5 courses
#     # st.table(top_5_courses_by_category)
#     st.write(top_5_courses_by_category)




#     # Sidebar for category selection
#     st.header("Udemy Courses Analysis: Best Courses")

#     # Ensure relevant columns are numeric
#     cleaned_data['Course Rating'] = pd.to_numeric(cleaned_data['Course Rating'], errors='coerce')

#     # Selectbox for category selection (unique ID for each widget)
#     category_top_courses = st.selectbox("Select a Category for Top Courses", cleaned_data['Category'].unique(), key="best_courses")

#     # Filter data by selected category
#     filtered_df = cleaned_data[cleaned_data['Category'] == category_top_courses]

#     # Find the best course by level (highest rating within each level)
#     best_courses = filtered_df.loc[filtered_df.groupby('Course Level')['Course Rating'].idxmax()]

#     # Create the bar chart
#     fig4 = px.bar(
#         best_courses,
#         x='Course Name',  # Best course name
#         y=['Course_Duration', 'Number of Lectures'],  # Y-axis values
#         color='Course Level',  # Grouped by levels
#         barmode='group',
#         title=f"Best Courses in {category_top_courses} by Level",
#         labels={
#             'Course Title': 'Course Name',
#             'value': 'Metrics',
#             'variable': 'Metrics (Duration or Lectures)',
#         },
#     )

#     # Display the chart
#     st.plotly_chart(fig4)

# if __name__ == "__main__":
#     main()


# # Load dataset
# @st.cache_data
# def load_data():
#     return pd.read_excel("Udemy Courses.xlsx")

# df = load_data()
# cleaned_data = preprocess_data("Udemy Courses.xlsx")

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(cleaned_data['Course Name'].fillna(''))

# # Function to recommend courses based on user input
# def recommend_courses(user_input, df, vectorizer, tfidf_matrix):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    
#     sim_scores = list(enumerate(cosine_sim[0]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:5]
    
#     recommended_courses = df.iloc[[i[0] for i in sim_scores]][['Course Name', 'Category', 'Course Rating']]
#     return recommended_courses



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from generate_embeddings import save_course_embeddings
# from preprocessing import preprocess_data  # Import the preprocessing function
# # from sklearn.feature_extraction.text import TfidfVectorizer  # Example method for generating embeddings
# from sentence_transformers import SentenceTransformer, util
# import joblib
# import os

# def load_cleaned_data():
#     file_path = 'C:/Users/Oumayma/Desktop/Studies/3EB/udemy/cleaned_data.pkl'
#     if os.path.exists(file_path):
#         return joblib.load(file_path)
#     else:
#         print(f"File {file_path} not found. Preprocessing data now.")
#         return None

# @st.cache_data
# def load_precomputed_embeddings():
#     embeddings = save_course_embeddings()
#     return embeddings

# @st.cache_resource
# def load_model():
#     return SentenceTransformer('all-MiniLM-L6-v2')

# @st.cache_data
# def compute_user_similarity(_user_input, _course_embeddings, _model):
#     """Compute cosine similarity between user input and course embeddings."""
#     user_embedding = _model.encode(_user_input, convert_to_tensor=True)
#     return util.cos_sim(user_embedding, _course_embeddings)[0]

# def main():
#     st.set_page_config(
#         page_title="Udemy Courses Data Analysis",
#         page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAA7UlEQVR4AWP4TwMwQIaOGvrl5d+tKZ9XeXx8feM3dQw9PuHrHOH3CzU/ABGQsaf8y8/P/8g39OWV3/PlQMYByQ2Rn3YVfYaYDiRv7/pJpqHX1v+AmAI34tm5XyscPgIFj3Z/JdPQK6u+QzVjind+HfZJ6ujRowUFBRUVFciCU6dOBQquWLGCTEOXLl3KAAbIgv7+/kCRhoYGMg0FOgfT0NDQUCobOmqop6cnxFDykxTEUCADInLv3j2ICDBhkJ/4GWAgLi4OmDzh3C9fvpBv6MuXL2VkZBhQwdWrV6mQTU+fPr0UDIDhMIzqqFFDAUfsdt6zPZyZAAAAAElFTkSuQmCC",
#         layout="wide",
#     )

#     st.markdown(
#         """
#         <style>
#             .stApp {
#                 background-color: whitesmoke;
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 flex-direction: column;
#             }
#             .css-18e3th9 {
#                 background-color: #A435F0 !important;
#                 color: black !important;
#                 font-size: 20px;
#                 text-align: center;
#             }
#             h1, h2, h3, h4, h5, h6, p {
#                 color: black !important;
#                 text-align: center;
#             }
#             .block-container {
#                 max-width: 1000px;
#                 margin: auto;
#             }
#             [data-testid="stSidebar"] {
#                 background-color: #A435F0;
#             }
#             [data-testid="stSidebar"] .css-18e3th9 {
#                 color: black;
#             }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     st.image("https://frontends.udemycdn.com/frontends-marketplace-experience/staticx/udemy/images/v7/logo-udemy.svg", width=150)
#     st.title("Udemy Courses Data Analysis and AI-Powered Recommendation System")

#     cleaned_data = load_cleaned_data()
#     if cleaned_data is None:
#         print("Preprocessing data...")
#         cleaned_data = preprocess_data("Udemy Courses.xlsx")
#         print("Saving cleaned data...")
#         joblib.dump(cleaned_data, "cleaned_data.pkl")

#     print("Proceeding with further steps using the cleaned data...")

#     course_embeddings = load_precomputed_embeddings()
#     model = load_model()

#     st.header("Course Recommendation System")
#     user_input = st.text_input("Tell us about your interest or field of study (e.g., 'I study computer science'): ")

#     if user_input:
#         with st.spinner("Finding the best courses for you..."):
#             similarities = compute_user_similarity(user_input, course_embeddings, model)
#             top_k = similarities.topk(k=5)

#         recommended_courses = cleaned_data.iloc[top_k.indices.tolist()][['Course Name', 'Category', 'Course Rating']]

#         if not recommended_courses.empty:
#             st.subheader(f"Recommended Courses for '{user_input}'")
#             st.write(recommended_courses)

#             filtered_data = cleaned_data[cleaned_data['Course Name'].isin(recommended_courses['Course Name'])]

#             st.subheader("Category Distribution")
#             category_counts = filtered_data['Category'].value_counts()
#             fig2 = px.pie(
#                 names=category_counts.index,
#                 values=category_counts.values,
#                 title="Category Distribution for Recommended Courses"
#             )
#             st.plotly_chart(fig2)

#             st.subheader("Course Duration vs. Ratings for Recommended Courses")
#             if 'Course_Duration' in filtered_data.columns and 'Course Rating' in filtered_data.columns:
#                 fig3 = px.scatter(
#                     filtered_data,
#                     x='Course_Duration',
#                     y='Course Rating',
#                     size='Course_Duration',
#                     color='Category',
#                     hover_data=['Course Name'],
#                     title="Course Duration vs. Course Ratings"
#                 )
#                 st.plotly_chart(fig3)
#         else:
#             st.write("No courses found matching your interest. Try another phrase!")


# if __name__ == "__main__":
#     main()


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
    file_path = 'C:/Users/Oumayma/Desktop/Studies/3EB/udemy/cleaned_data.pkl'
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
        joblib.dump(cleaned_data, "C:/Users/Oumayma/Desktop/Studies/3EB/udemy/cleaned_data.pkl")

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

    col1, col2 = st.columns([1, 1])  # Create two equal-width columns

    # Get Recommendations Button
    with col2:
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
    with col1:
        if st.button("Reset"):
            st.session_state["reset_count"] += 1  # Increment the reset counter
            st.experimental_rerun()


if __name__ == "__main__":
    main()
