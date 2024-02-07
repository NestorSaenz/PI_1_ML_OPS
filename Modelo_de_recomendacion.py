import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


archivo = 'Dataset/data_final.parquet'
df = pd.read_parquet(archivo)
df = df[['item_id', 'title', 'genres', 'playtime_2weeks', 'recommend', 'sentiment_analysis', ]]

# Filtramos por "recommend".
df = df[df['recommend'] == True]

# Filtramos por análisis de sentimiento.
df = df[df['sentiment_analysis'] == 2]

# Borramos la columna "recommend", ya que lo va vamos a usar.
df = df.drop(['recommend'], axis=1)

"""Sumamos la cantidad de horas jugadas por juego en las últimas dos semanas y la cantidad de recoendaciones positivas, agrupado por juego. Tomamos la cantidad jugada en las últimas dos semanas y lo usamos para el modelo de recomendación, entendiendo que esta cantidad de horas da una tendencia para poder recomendar. No tenemos demasiados parámetros para comparar similitud de coseno, por lo que una tendencia puede ser, en algunos casos, una buena manera de recomendar."""

df = df.groupby(['item_id', 'title', 'genres'], as_index=False).agg({
    'sentiment_analysis': 'count',
    'playtime_2weeks': 'sum',
 })

# Ordenamos de manera descendente en función del análisis de sentimiento.
df = df.sort_values(by='sentiment_analysis', ascending=False)
df = df.groupby(['item_id', 'title']).agg({
    'genres': lambda x: list(x),
    'sentiment_analysis': 'sum',
    'playtime_2weeks': 'sum'
}).reset_index()

"""Creamos una función que recibe un id como argumento (int) y devuelve una lista con 5 juegos recomendados basados en la similitud de coseno."""

def recomendacion_juego(id_juego):
    # Verifica si existe el id.
    if id_juego not in df['item_id'].values:
        return "ID de juego no encontrado"
    # Vectoriza (convierte texto en valores numéricos).
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # Convierte 'genres' a cadena y lo codifica con LabelEncoder.
    label_encoder = LabelEncoder()
    df['genres_str'] = label_encoder.fit_transform(df['genres'].astype(str))
    # Combina 'genres_str', 'title', 'sentiment_analysis', y 'playtime_2weeks' en una nueva columna.
    # Esto es para generar vectores y comparar cosenos.
    df['combined_features'] = (
        df['genres_str'].astype(str) + ' ' +
        df['title'] + ' ' +
        df['sentiment_analysis'].astype(str) + ' ' +
        df['playtime_2weeks'].astype(str)
    )
    # Aplica el vectorizador a la nueva columna.
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
    # Escala 'sentiment_analysis' y 'playtime_2weeks'.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['sentiment_analysis', 'playtime_2weeks']])
    # Agrega las características escaladas.
    tfidf_matrix = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(scaled_features)], axis=1)
    # Calcula la similitud de coseno entre los juegos.
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    # Obtiene el índice del juego en el DataFrame.
    idx = df[df['item_id'] == id_juego].index[0]
    # Obtiene las similitudes de coseno para el juego especificado.
    cosine_scores = list(enumerate(cosine_similarities[idx]))
    # Ordena los juegos por similitud de coseno. Cuanto más cercana a 1, más "parecido" es.
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
    # Obtiene los índices de los 5 juegos recomendados (excluyendo el juego actual) por similitud.
    recommended_indices = [i[0] for i in cosine_scores[1:6]]  
    # Obtiene los títulos de los juegos recomendados.
    recommended_titles = df['title'].iloc[recommended_indices].tolist()
    return recommended_titles


# Probamos la función.
recomendacion_juego(10)



