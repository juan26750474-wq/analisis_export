import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import mysql.connector

def run_anomaly_detection():
    # 1. CREDENCIALES (GitHub Secrets)
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASS')
    host = os.getenv('DB_HOST')
    name = os.getenv('DB_NAME')

    # 2. EXTRACCIÓN DE DATOS
    # Filtramos por los 5 productos clave
    query = """
    SELECT ano, mes, flujo, PA, codigo, peso, valor 
    FROM datos 
    WHERE codigo LIKE '0702%' -- Tomate
       OR codigo LIKE '0707%' -- Pepino
       OR codigo LIKE '070960%' -- Pimiento
       OR codigo LIKE '070993%' -- Calabacín
       OR codigo LIKE '070930%' -- Berenjena
    """
    
    try:
        # Usamos engine de SQLAlchemy para facilitar la lectura y escritura
        engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{name}")
        df = pd.read_sql(query, engine)
        print(f"✅ Datos importados: {len(df)} registros.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    if df.empty: return

    # 3. INGENIERÍA DE VARIABLES (Análisis de tendencias)
    df = df.sort_values(by=['codigo', 'PA', 'ano', 'mes'])
    df['precio_u'] = df['valor'] / df['peso']

    # Calculamos medias móviles de 12 meses para PESO, VALOR y PRECIO
    # Esto permite al modelo saber qué es "normal" para ese mercado
    for col in ['peso', 'valor', 'precio_u']:
        df[f'media_12m_{col}'] = df.groupby(['codigo', 'PA'])[col].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        # Ratio respecto a la media: si es 1 es normal, si es 10 es una anomalía masiva
        df[f'ratio_{col}'] = df[col] / (df[f'media_12m_{col}'] + 1e-6)

    # Limpieza de datos
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['precio_u', 'ratio_peso', 'ratio_valor'])

    # 4. MODELO ISOLATION FOREST (Multivariable)
    # Aquí es donde ocurre la magia: mira el cruce de todas estas variables
    features = ['peso', 'valor', 'precio_u', 'ratio_peso', 'ratio_valor', 'ratio_precio_u']
    
    # Contaminación al 1%: buscará el 1% más extraño combinando kilos, valor y precio
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_signal'] = model.fit_predict(df[features])

    # 5. FILTRAR Y VOLVER A LA BASE DE DATOS
    anomalias = df[df['anomaly_signal'] == -1].copy()

    if not anomalias.empty:
        # Quitamos las columnas de cálculo intermedio para que la tabla sea limpia
        cols_finales = ['ano', 'mes', 'flujo', 'PA', 'codigo', 'peso', 'valor', 'precio_u', 'anomaly_signal']
        
        # Guardar en phpMyAdmin (creará la tabla 'alertas_expor' si no existe)
        anomalias[cols_finales].to_sql('alertas_expor', con=engine, if_exists='replace', index=False)
        
        print(f"✅ Se han detectado {len(anomalias)} anomalías.")
        print("✅ Los resultados ya están disponibles en la tabla 'alertas_expor' de tu base de datos.")
    else:
        print("No se detectaron anomalías con los parámetros actuales.")

if __name__ == "__main__":
    run_anomaly_detection()
