# import enum
# import pandas as pd

# # Define an Enum for Course Levels
# class CourseLevel(enum.Enum):
#     ALL_LEVELS = 1
#     BEGINNER = 2
#     INTERMEDIATE = 3
#     EXPERT = 4

# def preprocess_data(file_path):
#     # Read the Excel file
#     df = pd.read_excel(file_path)

#     # Drop duplicates and clean column names
#     df = df.drop_duplicates()
#     df.columns = [col.strip() for col in df.columns]  # Strip extra spaces from column names

#     # Remove empty columns before filling missing values
#     empty_columns = df.columns[df.isnull().all()].tolist()
#     if empty_columns:
#         print(f"Dropping empty columns: {empty_columns}")
#         df = df.drop(columns=empty_columns)

#     # Fill missing values with "N/A" after removing empty columns
#     df = df.fillna("N/A")

#     # Convert 'Number of Lectures' to integer
#     if 'Number of Lectures' in df.columns:
#         df['Number of Lectures'] = (
#             df['Number of Lectures']
#             .str.extract(r'(\d+)')
#             .astype(float)
#             .fillna(0)
#             .astype(int)
#         )

#     # Map 'Course Level' to enum values
#     # if 'Course Level' in df.columns:
#     #     level_mapping = {
#     #         "All Levels": CourseLevel.ALL_LEVELS.value,
#     #         "Beginner": CourseLevel.BEGINNER.value,
#     #         "Intermediate": CourseLevel.INTERMEDIATE.value,
#     #         "Expert": CourseLevel.EXPERT.value,
#     #     }
#     #     df['Course Level'] = df['Course Level'].map(level_mapping).fillna(0).astype(int)

#     # Convert 'Course_Duration' to integer
#     if 'Course_Duration' in df.columns:
#         df['Course_Duration'] = (
#             df['Course_Duration']
#             .str.extract(r'(\d+)')
#             .astype(float)
#             .fillna(0)
#             .astype(int)
#         )
    
#     # if 'Course Rating' in df.columns:
#     #     df['Course Rating'] = (
#     #         df['Course Rating']
#     #         .str.extract(r'(\d+)')
#     #         .astype(float)
#     #         .fillna(0)
#     #         .astype(int)
#     #     )
#     if 'Course Rating' in df.columns:
#         df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')


#     # Convert numeric columns
#     numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     return df

# # Example of applying groupby and converting 'Course_Duration'
# def group_and_process(file_path):
#     df = preprocess_data(file_path)

#     # Group by 'Course Level' and aggregate
#     grouped_df = df.groupby('Course Level').agg({
#         'Course_Duration': 'mean',  # Example aggregation
#         'Number of Lectures': 'mean'  # Example aggregation
#     }).reset_index()

#     # Convert 'Course_Duration' to float
#     if 'Course_Duration' in df.columns:
#         df['Course_Duration'] = (
#             df['Course_Duration']
#             .str.extract(r'(\d+(\.\d+)?)')  # Adjusted to capture floating-point numbers
#             .astype(float)  # Convert directly to float
#             .fillna(0)  # Fill missing values with 0
#         )
#     print("Grouped and Processed DataFrame:")
#     print(grouped_df.head())

# # Example test function
# def test_group_and_process():
#     test_file_path = "test_Udemy_Courses.xlsx"  # Replace with a path to a test Excel file
#     print("Testing group and process...")
#     group_and_process(test_file_path)

# if __name__ == "__main__":
#     test_group_and_process()
# # Example test function
# def test_preprocessing():
#     # Path to the test file
#     test_file_path = "test_Udemy_Courses.xlsx"  # Replace with a path to a test Excel file

#     print("Testing preprocessing...")
#     df = preprocess_data(test_file_path)

#     print("Processed DataFrame:")
#     print(df.head())

# if __name__ == "__main__":
#     test_preprocessing()



import enum
import pandas as pd

# Define an Enum for Course Levels
class CourseLevel(enum.Enum):
    ALL_LEVELS = 1
    BEGINNER = 2
    INTERMEDIATE = 3
    EXPERT = 4

def preprocess_data(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Drop duplicates and clean column names
    df = df.drop_duplicates()
    df.columns = [col.strip() for col in df.columns]  # Strip extra spaces from column names

    # Remove empty columns before filling missing values
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        print(f"Dropping empty columns: {empty_columns}")
        df = df.drop(columns=empty_columns)

    # Fill missing values with "N/A" after removing empty columns
    df = df.fillna("N/A")

    # Convert 'Number of Lectures' to integer
    if 'Number of Lectures' in df.columns:
        df['Number of Lectures'] = (
            df['Number of Lectures']
            .str.extract(r'(\d+)')
            .astype(float)
            .fillna(0)
            .astype(int)
        )

    # Map 'Course Level' to enum values
    if 'Course Level' in df.columns:
        level_mapping = {
            "All Levels": CourseLevel.ALL_LEVELS.value,
            "Beginner": CourseLevel.BEGINNER.value,
            "Intermediate": CourseLevel.INTERMEDIATE.value,
            "Expert": CourseLevel.EXPERT.value,
        }
        df['Course Level'] = df['Course Level'].map(level_mapping).fillna(0).astype(int)

    # Convert 'Course_Duration' to integer
    if 'Course_Duration' in df.columns:
        df['Course_Duration'] = (
            df['Course_Duration']
            .str.extract(r'(\d+)')
            .astype(float)
            .fillna(0)
            .astype(int)
        )

    # Convert 'Course Rating' to numeric
    if 'Course Rating' in df.columns:
        df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')

    # Save the cleaned DataFrame as a pickle file
    df.to_pickle("cleaned_data.pkl")
    print("cleaned_data.pkl saved.")
    return df

# Test function
def test_preprocessing():
    # Path to the test file
    test_file_path = "Udemy Courses.xlsx"  # Replace with a path to a test Excel file
    output_file_path = "cleaned_data.pkl"

    print("Testing preprocessing...")
    preprocess_data(test_file_path, output_file_path)

    # Load and verify the saved pickle file
    loaded_df = pd.read_pickle(output_file_path)
    print("Loaded DataFrame from pickle:")
    print(loaded_df.head())

if __name__ == "__main__":
    test_preprocessing()
