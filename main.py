from fastapi import FastAPI
import pandas as pd


app = FastAPI()


@app.get('/')
async def my_function():
    return 'PROYECTO INDIVIDUAL Nº1 Machine Learning Operations (MLOps)'

@app.get("/developer/{desarrollador}")
async def desarrollador(desarrollador:str):
    df1 = pd.read_parquet('Dataset/steam_games.parquet')
    df_desarrollador = df1[df1['developer'] == desarrollador.capitalize()]
    df_resultado = pd.DataFrame()
    df_resultado['Años'] = df_desarrollador['release_date'].unique()
    df_resultado['Cantidad de items'] = df_desarrollador.groupby('release_date').size().values
    df_resultado['Porcentaje free'] = (len(df_desarrollador[df_desarrollador['price'] == 0]) / df_resultado['Cantidad de items'].sum()) * 100
    resultado_dict = df_resultado.to_dict(orient='records')
    return resultado_dict

@app.get('/User_id/{user_id}')
async def userdata(user_id: str):
    df2 = pd.read_parquet('Dataset/endpoint_2_copia.parquet')
    df_user_id = df2[df2['user_id'] == user_id.capitalize()]
    dinero_gastado= (df_user_id['price'].sum())
    porcentaje_recomen = len(df_user_id['recommend']=='True')/len(df_user_id)*100
    cant_items = df_user_id.iloc[0,3].astype(float)
    
    return {'usuario':user_id, 'Dinero gastado':dinero_gastado, 'porcentaje de recomendación':porcentaje_recomen, 'Cantidad de items': cant_items}

@app.get('/Genero/{genero}')
async def UserForGenre(genero: str):
    df3 = pd.read_parquet('Dataset/endpoint_3_copia.parquet')
    data = df3[df3['genres'] == genero.capitalize()]
    usuario_horas = data.groupby('user_id')['playtime_forever'].sum().idxmax(0)
    lista_horas = data.groupby('release_date')['playtime_forever'].sum().reset_index() 
    resultado ={'Usuario con más horas jugadas para ' + genero: usuario_horas,
              'Horas jugadas': [{'Año': int(row['release_date']),  'Horas':int(row['playtime_forever'])} for index, row in lista_horas.iterrows()]
  }
    return resultado

@app.get('/Año')
async def best_developer_year(año: int ): 
    df4 = pd.read_parquet('Dataset/endpoint_4_5_copia.parquet')
    df4['release_date'] = df4['release_date'].astype(int)
    data = df4[df4['release_date']== año]
    data = data[(data['recommend'] == True) & (data['sentiment_analysis'] == 2)].sort_values(by= 'developer',ascending= False)
    df_dvelopers = data.groupby('developer')['sentiment_analysis'].sum().index[:3]
    lista = df_dvelopers.to_list()
    resultado_dict = {'Primer puesto': lista[0], 'Segundo puesto': lista[1], 'Tercer puesto': lista[2]}
   
    
    return resultado_dict   


    
@app.get('/Desarrolladora/{desarrolladora}')
async def developer_reviews_analysis(desarrolladora: str ):
    df = pd.read_parquet('Dataset/endpoint_4_5.parquet')
    df_desarrolladora = df[df['developer'] == desarrolladora]
    positivos = (df_desarrolladora['sentiment_analysis'] == 2).count()
    negativos = (df_desarrolladora['sentiment_analysis'] == 0).count()
    return {desarrolladora:f'[Negative = {negativos}, Positive = {positivos}]'}

     
    
    
    