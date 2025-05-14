import pandas as pd
import numpy as np
import ast  
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("bollywood_movies_dataset.csv", low_memory=False)
movies = df[['Movie Name', 'Rating(10)', 'Genre']].copy()
print(movies.columns)
filtered_df = df.dropna(subset=['Rating(10)', 'Genre'])
filtered_df['genre_list'] = filtered_df['Genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(filtered_df['genre_list']),
                             columns=mlb.classes_, index=filtered_df.index)

features = pd.concat([filtered_df[['Rating(10)']], genre_encoded], axis=1)

scaler = StandardScaler()
final_features = scaler.fit_transform(features)

inertia_values = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(final_features)
    inertia_values.append(model.inertia_)
    silhouette_scores.append(silhouette_score(final_features, model.labels_))

final_model = KMeans(n_clusters=4, random_state=42)
filtered_df['Cluster'] = final_model.fit_predict(final_features)

print(filtered_df[['Movie Name', 'Rating(10)', 'Genre', 'Cluster']].head(10))
def parse_genres(genre_str):
    if isinstance(genre_str, str):
        return [g.strip() for g in genre_str.split(',') if g.strip()]
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []


movies['genre_list'] = movies['Genre'].apply(parse_genres)

inertia_values = []
silhouette_scores = []
k_values = range(2, 21)

for k in k_values:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(final_features)
    inertia_values.append(kmeans_model.inertia_)
    silhouette_scores.append(silhouette_score(final_features, kmeans_model.labels_))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].plot(k_values, inertia_values, marker='o')
axs[0].set_title('Elbow Method for Optimal k')
axs[0].set_xlabel('Number of clusters (k)')
axs[0].set_ylabel('Inertia')
axs[0].grid(True)

axs[1].plot(k_values, silhouette_scores, marker='s', color='green')
axs[1].set_title('Silhouette Scores for Different k')
axs[1].set_xlabel('Number of clusters (k)')
axs[1].set_ylabel('Silhouette Score')
axs[1].grid(True)

plt.tight_layout()
plt.show()
genre_counts = movies['genre_list'].explode().value_counts().sort_values(ascending=False)
genre_counts = pd.Series([genre for sublist in movies['genre_list'] for genre in sublist]).value_counts()

movies = movies.dropna(subset=['genre_list'])
genre_counts = movies['genre_list'].explode().value_counts().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='rainbow')
plt.title('Genre Distribution')
plt.xticks(rotation=45)
plt.ylabel('Number of Movies')
plt.xlabel('Genre')
plt.tight_layout()
plt.show()

#plt.figure(figsize=(12, 6))
#sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='rainbow')
#plt.title('Genre Distribution')
#plt.xticks(rotation=45)
#plt.ylabel('Number of Movies')
#plt.xlabel('Genre')
#plt.tight_layout()
#plt.show()
lb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(movies['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
movies['genre_list'] = movies['Genre'].apply(
    lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
)

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(movies['genre_list']), columns=mlb.classes_, index=movies.index)
combined = pd.concat([movies[['Rating(10)']], genre_encoded], axis=1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(combined)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(movies[['Rating(10)']])
scaled_df = pd.DataFrame(scaled_features, columns=['Rating(10)'], index=movies.index)

final_features = pd.concat([genre_df, scaled_df], axis=1)
optimal_k = 10
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
movies['cluster'] = kmeans.fit_predict(final_features)

plt.figure(figsize=(10, 5))
sns.countplot(x=movies['cluster'], palette='rainbow')
plt.title('Number of Movies per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
def recommend_movies(movie_title, n=5):
    """
    Recommend similar movies based on genre, popularity, and rating using clustering.
    
    Parameters:
        movie_title (str): Title of the movie to base recommendations on.
        n (int): Number of similar movies to return.
        
    Returns:
        pd.DataFrame: Top N recommended movies in the same cluster.
    """
    
    target = movies[movies['Movie Name'].str.lower() == movie_title.lower()]
    
    if target.empty:
        return f" Movie titled '{movie_title}' not found in dataset."
  
    cluster_id = target['cluster'].values[0]
    
    
    similar_movies = movies[(movies['cluster'] == cluster_id) & 
                            (movies['Movie Name'].str.lower() != movie_title.lower())]


    return similar_movies[['Movie Name','Rating(10)', 'Genre']].sort_values(
        by=['Genre'], ascending=False).head(n)

print(" Recommended movies similar to 'Murder 2':")
print(recommend_movies("Murder 2", n=5))