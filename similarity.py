import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')

def preprocess_data(dataframe, text_column):
    stop_words = set(stopwords.words('english'))

    # Ensure the text_column exists in the dataframe
    if text_column not in dataframe.columns:
        raise ValueError(f"{text_column} does not exist in the dataframe")

    dataframe['clean_text'] = dataframe[text_column].apply(lambda x: ' '.join([word for word in str(x).lower().split() if word not in stop_words]))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))
    dataframe['clean_text'] = dataframe['clean_text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    
    return dataframe

def calculate_similarity(df, threshold, identifier_column, text_column, additional_columns):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i, j] > threshold:
                summary1_id = df[identifier_column].iloc[i]
                summary2_id = df[identifier_column].iloc[j]
                similarity = similarity_matrix[i, j]
                pairs.append((summary1_id, summary2_id, similarity))

    pairs_df = pd.DataFrame(pairs, columns=['Node 1', 'Node 2', 'Similarity'])

    if additional_columns:
        for col in additional_columns:
            temp_df1 = df[[identifier_column, col]].add_suffix(f'_Node1').rename(columns={f'{identifier_column}_Node1': 'Node 1'})
            temp_df2 = df[[identifier_column, col]].add_suffix(f'_Node2').rename(columns={f'{identifier_column}_Node2': 'Node 2'})
            pairs_df = pd.merge(pairs_df, temp_df1, on='Node 1', how='left')
            pairs_df = pd.merge(pairs_df, temp_df2, on='Node 2', how='left')

    return pairs_df
