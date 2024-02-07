![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)
![Numpy](https://img.shields.io/badge/-Numpy-333333?style=flat&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-333333?style=flat&logo=matplotlib)
![Scikitlearn](https://img.shields.io/badge/-Scikitlearn-333333?style=flat&logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/-FastAPI-333333?style=flat&logo=fastapi)
![Render](https://img.shields.io/badge/-Render-333333?style=flat&logo=render)

 # PI_1_ML_OPS

Este proyecto simula el rol de un MLOps Engineer, es decir, la combinación de un Data Engineer y Data Scientist, para la plataforma multinacional de videojuegos Steam. Para su desarrollo, se entregan unos datos y se solicita un Producto Mínimo Viable que muestre una API deployada en un servicio en la nube y la aplicación de un modelo de Machine Learning para hacer recomendaciones de juegos.

# Introducción

Para este proyecto se nos proporciona un conjunto de tres archivos en formato JSON: de steam (Steam es una plataforma de distribución digital de videojuegos desarrollada por Valve Corporation) para poder trabajar en ellos y crear un Producto Minimo Viable (MVP), que contiene una la implementaciónde una API  y con un modelo de Machine Learning. los datos provienen de los archivos siguientes: 

  
*  **steam_games* información  relacionada a los juegos dentro de la plataforma Steam. Por ejemplo: Nombre del juego, género, fecha de lanzamiento, entre otras. 

  
* **user_reviews* información que detalla las reseñas realizadas por los usuarios de la plataforma Steam. 

  
* **user_items* información acerca de la actividad de los usuarios dentro de la plataforma Steam.

## ETL
Se realizó la extracción, transformación y carga (ETL) de los tres conjuntos de datos entregados.
En esta fase del proyecto se realiza la extracción de datos, a fin de familiarizarse con ellos y comenzar con la etapa de limpieza de datos que nos permita el correcto entedimiento. Terminada la limpieza se generará el conjunto de datos para la siguiente fase, estos se guardaron en formato parquet. 


Los detalles del ETL para cada Dataset se puede ver en [ETL](https://github.com/NestorSaenz/PI_1_ML_OPS/tree/main/ETL)
  
## Feature engineering
En esta etapa se realizo el analisis de sentimientos a los reviews de los usuarios. Para ello se creó una nueva columna llamada 'sentiment_analysis' que reemplaza a la columna que contiene los reviews donde clasifica los sentimientos de los comentarios con la siguiente escala:

* 0 si es malo,
* 1 si es neutral o esta sin review
* 2 si es positivo.

Todos los detalles del desarrollo se pueden ver en la Jupyter Notebook [ETL](./ETL/users_review.ipynb)



 
