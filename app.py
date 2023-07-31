import streamlit as st
import pandas as pd
import difflib
import matplotlib.pyplot as plt
import sqlite3
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variable to store recommendations
recommend = []

def load_data():
    # loading the data from the csv file to pandas dataframe
    hobby_data = pd.read_excel('/Users/zulhilmi/Documents/FYP/TestFYP/hobby.xlsx', sheet_name='hobby')
    return hobby_data

def load_feedback_data():
    # Connect to the SQLite database
    conn = sqlite3.connect('userfeedback.db')
    
    # Read the feedback data from the 'sus_responses' table into a DataFrame
    feedback_data = pd.read_sql_query("SELECT * FROM sus_responses", conn)

    # Close the connection
    conn.close()

    # Drop the 'Timestamp' column
    feedback_data.drop(columns=['timestamp'], inplace=True)
    
    return feedback_data

def calculate_raw_score(row):
    raw_score = 0
    for i in range(len(row)):
        if i % 2 == 0:  # Odd-numbered question
            odd_score = row[i] - 1
            raw_score += odd_score
        else:  # Even-numbered question
            even_score = 5 - row[i]
            raw_score += even_score
    return raw_score

def calculate_final_sus_score(feedback_data):
    # Modify this based on your actual column names in the feedback data
    sus_question_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
    
    # Calculate the raw scores for each respondent
    feedback_data['SUS_Raw_Score'] = feedback_data[sus_question_columns].apply(calculate_raw_score, axis=1)

    # Calculate the final SUS score
    feedback_data['Final_SUS_Score'] = feedback_data['SUS_Raw_Score'] * 2.5

    return feedback_data

def calculate_average_sus_score(feedback_data):
    return feedback_data['Final_SUS_Score'].mean()

# Function to insert the evaluation responses into the existing table
def insert_response(responses):
    # Connect to the existing database
    conn = sqlite3.connect('userfeedback.db')
    c = conn.cursor()

    # Insert the evaluation responses into the table
    c.execute('''INSERT INTO sus_responses (Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', tuple(responses.values()))

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def find_recommendations(hobby_name, hobby_data):
    # selecting the relevant features for recommendation
    selected_features = ['type', 'equipment', 'location', 'versatility', 'pace']

    # replacing the null values with null string
    for feature in selected_features:
        hobby_data[feature] = hobby_data[feature].fillna('')

    # combining all the 5 selected features
    combined_features = hobby_data['type'] + ' ' + hobby_data['equipment'] + ' ' + hobby_data['versatility'] + ' ' + hobby_data['pace']

    # converting the text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)

    list_of_all_name = hobby_data['name'].tolist()
    find_close_match = difflib.get_close_matches(hobby_name, list_of_all_name)
    close_match = find_close_match[0]
    index_of_the_hobby = hobby_data[hobby_data.name == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_hobby]))
    sorted_similar_hobby = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    top_10_hobbies = []
    skip_first_hobby = True

    for hobby in sorted_similar_hobby:
        index = hobby[0]
        title_from_index = hobby_data[hobby_data.index == index]['name'].values[0]
        details_from_index = hobby_data[hobby_data.index == index]['details'].values[0]
        images_from_index = hobby_data[hobby_data.index == index]['images'].values[0]
        equipment_from_index = hobby_data[hobby_data.index == index]['equipment'].values[0] 
        
        # Skip the first hobby
        if skip_first_hobby:
            skip_first_hobby = False
            continue
        
        top_10_hobbies.append({'name': title_from_index, 'details': details_from_index, 'images': images_from_index, 'equipment': equipment_from_index})
        if len(top_10_hobbies) == 10:
            break

    return top_10_hobbies


def find_fuzzy_recommendations(user_input, hobby_data):
    user_input_lower = user_input.lower()

    # Use fuzzywuzzy to calculate similarity scores (case-insensitive)
    similarity_scores = [(index, fuzz.partial_ratio(user_input_lower, name.lower())) for index, name in enumerate(hobby_data['name'])]
    sorted_similar_hobby = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_10_hobbies = []
    skip_first_hobby = True

    for index, similarity_scores in sorted_similar_hobby:
        title_from_index = hobby_data.loc[index, 'name']
        details_from_index = hobby_data.loc[index, 'details']
        images_from_index = hobby_data.loc[index, 'images']
        equipment_from_index = hobby_data.loc[index, 'equipment']

        # Check if the first hobby exactly matches the user input, then skip it
        if skip_first_hobby and title_from_index.lower() == user_input_lower:
            skip_first_hobby = False
            continue

        top_10_hobbies.append({'name': title_from_index, 'details': details_from_index, 'images': images_from_index, 'equipment': equipment_from_index})
        if len(top_10_hobbies) == 10:
            break

    return top_10_hobbies


def get_recommendations(user_preferences):
    # Load data
    df = pd.read_excel('hobby.xlsx')

    # Select the relevant columns for content-based filtering
    columns = ['type', 'location', 'versatility', 'pace', 'equipment', 'details']

    # Convert non-numeric columns to numeric using one-hot encoding
    df_encoded = pd.get_dummies(df[columns])

    # Preprocess the 'details' column using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['details'])

    # Convert the TF-IDF matrix to a DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Combine the one-hot encoded features with the TF-IDF features
    combined_df = pd.concat([df_encoded, tfidf_df], axis=1)

    # Create the user_df DataFrame
    user_df = pd.DataFrame([user_preferences], columns=columns)

    # Preprocess the user's 'details' input using TF-IDF
    user_tfidf_matrix = tfidf_vectorizer.transform(user_df['details'])
    user_tfidf_df = pd.DataFrame(user_tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Encode the user preferences as separate columns
    user_encoded = pd.get_dummies(user_df)
    # Combine the user's one-hot encoded features with the TF-IDF features
    user_combined_df = pd.concat([user_encoded, user_tfidf_df], axis=1)

    # Reindex user_combined_df to match the columns of combined_df
    user_combined_df = user_combined_df.reindex(columns=combined_df.columns, fill_value=0)

    # Calculate the cosine similarity between the user preferences and all hobbies
    cosine_similarities = cosine_similarity(user_combined_df, combined_df)

    # Get the indices of hobbies sorted by cosine similarity in descending order
    related_indices = cosine_similarities.argsort()[0][::-1]

    # Get the top 10 most similar hobbies
    top_10_hobbies = df.iloc[related_indices[:10], :]

    return top_10_hobbies[['name', 'type', 'equipment', 'location', 'versatility', 'pace', 'details', 'images']].to_dict('records')

def display_recommendations(recommendations):
    st.subheader("Top 10 recommended hobbies:")
    for index, hobby in enumerate(recommendations):
        with st.expander(f"{index+1}. {hobby['name']}"):

            # Display the image from the internet link in the 'image' column
            st.image(hobby['images'], use_column_width=True)

            st.write(f"Details: {hobby['details']}")

            # Generate the Shopee link based on the recommended hobby name
            shopee_link = f"https://shopee.com.my/search?keyword={hobby['name'].replace(' ', '-')}"
            st.write(f"Shopee Link: {shopee_link}")

            st.write("or")

            # Generate the Shopee link based on the recommended hobby name
            equipment_link = f"https://shopee.com.my/search?keyword={hobby['equipment'].replace(' ', '-')}"
            st.write(f"Shopee Link: {equipment_link}")

# Define the evaluation questions
evaluation_questions = [
    "Would you use the system frequently?",
    "The system unnecessarily complex.",
    "The system is easy to use?",
    "Do you think you need help from someone to use the system?",
    "All the features are well integrated into the system.",
    "The system is too confusing.",
    "Learning curve to use the system is easy.",
    "The system is hard to use.",
    "Feel confident using the system.",
    "Need to learn a lot before using the system.",
]

# Create a Pandas DataFrame to store the evaluation question responses
response_options = ["stronglyDisagree", "disagree", "neutral", "agree", "stronglyAgree"]
evaluation_data = pd.DataFrame(columns=["Question"] + response_options)

def main():
    
    # Set linear gradient background color
    st.markdown(
        """
        <style>

        .css-1avcm0n {
            background: #F7F1E9;
        }
        
        .css-1wrcr25 {
            background: linear-gradient(to bottom, rgba(247, 241, 233, 0.8), rgba(173, 182, 184, 0.8)),
                        url('https://previews.123rf.com/images/dimapolie/dimapolie1804/dimapolie180400066/98944772-vector-hobby-pattern-hobby-seamless-background.jpg');
            background-size: cover;
        }

        .css-10trblm {
            color: #415C6F;
        }

        .block-container {
            margin-top: 60px;
            padding: 20px;
            background-color: #dae2e3;
            border-radius: 10px;
            max-width: 55rem;
        }

        .css-729dqf > label {
            font-size: 18px !important;
        }

        .css-q8sbsg p {
            font-size: 24px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    hobby_data = load_data()

    # Load feedback data
    feedback_data = load_feedback_data()

    # Calculate final SUS score
    feedback_data = calculate_final_sus_score(feedback_data)

    st.title("HOBIST")
    sidebar_option = st.sidebar.radio("NAVIGATE", ("Home", "Enter Your Preferences", "Evaluation", "System Usability Scale"))

    if sidebar_option == "Home":
        with st.container():
            
            st.markdown(
                """
                <style>
                .paragraph {
                    text-align: justify;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<p class='paragraph'>Welcome to Hobist: hobby recommendation system! If you're looking for exciting new hobbies but do not know where to start, you have come to the right place to suggest hobbies based on your preferred features. Simply input your preferences, and our intelligent algorithm will analyze them using encoding features technique, matching them with our database of hobbies to provide you with personalized recommendations. Get ready to explore a world of new interests and passions.</p>", unsafe_allow_html=True)
            st.markdown("<p class='paragraph'>To get the gist of this recommendation system, below you can search for similar hobbies based on your current favorite.</p>", unsafe_allow_html=True)
            hobby_name = st.text_input("Please enter a hobby:", value="", key="hobby_input")

            # Find button
            if st.button("Find"):
                # Check if the user input is null or consists of only whitespace
                if not hobby_name.strip():
                    st.write("Please enter a hobby!")
                else:
                    # Convert user input and hobby names to lowercase for case-insensitive comparison
                    hobby_name_lower = hobby_name.lower()
                    hobby_data['name_lower'] = hobby_data['name'].str.lower()

                    # Check if the user input is available in the dataset (case-insensitive)
                    if hobby_name_lower in hobby_data['name_lower'].tolist():
                        recommendations = find_recommendations(hobby_name_lower, hobby_data)
                    else:
                        # If user input is not available in the dataset, use fuzzywuzzy
                        recommendations = find_fuzzy_recommendations(hobby_name_lower, hobby_data)

                    display_recommendations(recommendations)

    elif sidebar_option == "Enter Your Preferences":
        with st.container():
            st.write("We strive to provide you with personalized suggestions based on your preferences. To tailor the recommendations specifically to your liking, we may ask a few additional questions.")
            st.write("Enter your preferences below:")

            # User input for preferences
            col1, col2 = st.columns(2)

            # User input for type
            with col1:
                st.subheader("Type")
                type_options = ['Physical', 'Cerebral', 'Creativity', 'Community', 'Collectibles']
                user_preferences = {}
                user_preferences['type'] = st.radio("Select type:", type_options)

                # User input for location
                st.subheader("Location")
                location_options = ['Indoor', 'Outdoor']
                user_preferences['location'] = st.radio("Select location:", location_options)

            with col2:
                # User input for versatility
                st.subheader("Versatility")
                st.write("Do you like do things individually or with other people?")
                versatility_options = ['Individual', 'Pairs', 'Groups', 'Team']
                user_preferences['versatility'] = st.radio("Select versatility:", versatility_options)

                # User input for pace
                st.subheader("Pace")
                st.write("Do you like to relax or want some challange?")
                pace_options = ['Leisure', 'Exercise', 'Recreational', 'Competitive', 'Extreme']
                user_preferences['pace'] = st.radio("Select pace:", pace_options)

            # User input for equipment
            st.subheader("Equipment")
            st.write("Insert equipments that required in your desired activities (if any)")
            user_preferences['equipment'] = st.text_input("Enter equipment:")

            # User input for details
            st.subheader("Details")
            st.write("Insert extra details on activities you may like.")
            st.write("example: car racing karting race track")
            st.write("'.' is not required")
            user_preferences['details'] = st.text_input("Enter details:")

            if st.button("Get Recommendations"):
                top_hobbies = get_recommendations(user_preferences)
                display_recommendations(top_hobbies)

    elif sidebar_option == "Evaluation":
        with st.container():
            st.markdown(
                """
                <style>
                .paragraph {
                    text-align: justify;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.title("Users Evaluation")

            st.markdown("<p class='paragraph'>We're excited to help you discover new and exciting hobbies that match your interests and preferences. In order to provide you with the best recommendations possible, we would appreciate your feedback on the usability of our system. To do this, we have included the System Usability Scale (SUS) questionnaire. Thank you for taking the time to provide us with your feedback.</p>", unsafe_allow_html=True)
            
            # Create a dictionary to store responses
            responses = {}

            # Display each evaluation question and collect responses
            for i, question in enumerate(evaluation_questions):
                st.subheader(f"Question {i+1}")
                st.write(question)
                # Use st.columns to create a column for the slider
                col1 = st.columns(1)
                with col1[0]:
                    response = st.slider("Scale: 1 (Strongly Disagree) to 5 (Strongly Agree)", min_value=1, max_value=5, step=1, key=f"question_{i}_slider")
                    responses[f"Q{i+1}"] = response

            # Display the collected responses in 5 columns
            st.subheader("Evaluation Responses")
            columns = st.columns(5)

            # Loop through the responses and display them in the columns
            for i, (question, response) in enumerate(responses.items()):
                column_index = i % 5
                with columns[column_index]:
                    st.write(f"{question}: {response}")

            # Add a "Submit" button to store the responses into the database
            if st.button("Submit"):
                insert_response(responses)
                st.success("Evaluation responses submitted successfully!")

    elif sidebar_option == "System Usability Scale":
        with st.container():
            #st.write(feedback_data)

            # Set bin edges with class size 10
            bin_edges = list(range(0, 101, 10))

            # Display histogram of final SUS scores
            st.write("This is histogram of the system usability score.")
            # Calculate and display the average SUS score
            average_sus_score = calculate_average_sus_score(feedback_data)
            st.subheader(f"Average SUS Score: {average_sus_score:.2f}")
            plt.hist(feedback_data['Final_SUS_Score'], bins=bin_edges, rwidth=0.9, align='mid')
            # Set the x-axis ticks to start from 0 and increment by 10
            plt.xticks(range(0, 101, 10))
            plt.xlabel('Final SUS Score')
            plt.ylabel('Frequency')
            plt.title('Histogram of Final SUS Scores')
            st.pyplot(plt)
            
if __name__ == "__main__":
    main()