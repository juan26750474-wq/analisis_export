import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import mysql.connector

def run_anomaly_detection():
    # 1. CONEXIÓN A LA BASE DE DATOS (Usando Secrets/Environment Variables)
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASS'),
            database=os.getenv('DB_NAME')
        )
        print("✅ Conexión establecida con éxito.")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return

    # 2. EXTRACCIÓN DE DATOS FILTRADOS
    # Usamos prefijos comunes para: Tomate(0702), Pepino(0707), Pimiento(070960), Calabacín(070993), Berenjena(070930)
    query = """
    SELECT ano, mes, flujo, PA, codigo, peso, valor 
    FROM datos 
    WHERE codigo LIKE '0702%' 
       OR codigo LIKE '0707%' 
       OR codigo LIKE '070960%' 
       OR codigo LIKE '070993%' 
       OR codigo LIKE '070930%'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ No se encontraron datos para los criterios seleccionados.")
        return

    # 3. INGENIERÍA DE VARIABLES (FEATURE ENGINEERING)
    # Ordenar cronológicamente para cálculos temporales
    df = df.sort_values(by=['codigo', 'PA', 'ano', 'mes'])
    
    # Variable base: Precio Unitario
    df['precio_u'] = df['valor'] / df['peso']
    
    # Variable avanzada: Media móvil del precio unitario (últimos 12 meses registrados por producto/país)
    df['media_movil_12m'] = df.groupby(['codigo', 'PA'])['precio_u'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())
    
    # Variable de desviación: ¿Cuánto se aleja el precio actual de su media histórica?
    df['diff_media'] = df['precio_u'] - df['media_movil_12m']
    
    # Limpieza de valores infinitos o nulos (por divisiones por cero en peso)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['precio_u', 'diff_media'])

    # 4. MODELO ISOLATION FOREST
    # Seleccionamos las columnas que el modelo analizará
    # Analizamos peso, valor, precio_u y la diferencia con su media histórica
    features = ['peso', 'valor', 'precio_u', 'diff_media']
    
    # contamination=0.01 significa que esperamos un 1% de anomalías (ajustar según necesidad)
    model = IsolationForest(contamination=0.01, random_state=42)
    
    # Entrenar y predecir
    # 1 = normal, -1 = anomalía
    df['anomaly_signal'] = model.fit_predict(df[features])

    # 5. RESULTADOS
    anomalias = df[df['anomaly_signal'] == -1]
    
    print(f"--- Análisis Completado ---")
    print(f"Total registros analizados: {len(df)}")
    print(f"Anomalías detectadas: {len(anomalias)}")
    
    # Mostrar las 10 anomalías más claras
    if not anomalias.empty:
        print("\nPrincipales anomalías encontradas (Top 10):")
        print(anomalias[['ano', 'mes', 'PA', 'codigo', 'peso', 'valor', 'precio_u']].head(10))
        
        # Aquí podrías añadir el código para INSERTAR estas filas en una tabla nueva de tu DB
    
if __name__ == "__main__":
    run_anomaly_detection()
