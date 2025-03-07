import pandas as pd

def preprocess_data(df):

    df.drop_duplicates(inplace=True)

    df['title'].fillna('Unknown', inplace=True)
    df['release_date'].fillna('1900-01-01', inplace=True)  # Default date for missing values
    df['overview'].fillna('No overview available', inplace=True)
    df['tagline'].fillna('No tagline', inplace=True)
    df['genres'].fillna('[]', inplace=True)
    df['production_companies'].fillna('[]', inplace=True)
    df['production_countries'].fillna('[]', inplace=True)
    df['spoken_languages'].fillna('[]', inplace=True)
    df['keywords'].fillna('[]', inplace=True)
    df['homepage'].fillna('No homepage', inplace=True)
    df['imdb_id'].fillna('No ID', inplace=True)
    df['backdrop_path'].fillna('No image', inplace=True)
    df['poster_path'].fillna('No image', inplace=True)

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    categorical_cols = ['status', 'original_language', 'adult']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    threshold = 0.3  
    df = df.dropna(thresh=int(threshold * len(df)), axis=1)
    
    return df

df = pd.read_csv('TMDB_movie_dataset_v11.csv')


df = preprocess_data(df)

print(df.info())

df.to_csv('TMDB_movie_dataset_preprocessed.csv', index=False)

print("Preprocessed data saved as 'TMDB_movie_dataset_preprocessed.csv'")
