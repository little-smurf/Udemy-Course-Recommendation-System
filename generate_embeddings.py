from sentence_transformers import SentenceTransformer
import joblib
import os
import pandas as pd

def load_cleaned_data():
    """
    Load cleaned course data from a preprocessed file.
    Ensure the data contains a column 'Course Description' or 'Course Name' for embeddings.
    """
    cleaned_data_path = "cleaned_data.pkl"
    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Preprocessed data file '{cleaned_data_path}' not found.")

    cleaned_data = joblib.load(cleaned_data_path)
    if isinstance(cleaned_data, pd.DataFrame) and 'Course Description' in cleaned_data.columns:
        return cleaned_data['Course Description'].tolist()
    elif isinstance(cleaned_data, pd.DataFrame) and 'Course Name' in cleaned_data.columns:
        return cleaned_data['Course Name'].tolist()
    else:
        raise ValueError("Cleaned data must contain 'Course Description' or 'Course Name'.")

def generate_course_embeddings(courses):
    """
    Generate semantic embeddings for courses using SentenceTransformer.

    Args:
        courses (list): List of course descriptions or titles.

    Returns:
        np.ndarray: Array of course embeddings with shape (N, embedding_dim).
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(courses, convert_to_tensor=False)
    return embeddings

def save_course_embeddings():
    """
    Save or load course embeddings to/from a file.
    Ensures embeddings are generated using SentenceTransformer.
    """
    embeddings_file_path = "course_embeddings.pkl"

    if os.path.exists(embeddings_file_path):
        os.remove(embeddings_file_path)  # Delete existing incorrect embeddings

    print("Loading cleaned data from 'cleaned_data.pkl'...")
    courses = load_cleaned_data()

    print("Generating new course embeddings...")
    embeddings = generate_course_embeddings(courses)

    print(f"Saving generated embeddings to {embeddings_file_path}...")
    joblib.dump(embeddings, embeddings_file_path)
    return embeddings

if __name__ == "__main__":
    embeddings = save_course_embeddings()
    print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
