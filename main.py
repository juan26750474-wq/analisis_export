import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

def run_top_market_analysis():
    engine = create_engine(f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")

    # 1. EXTRACCIÓN DE DATOS BASE
    query = """
    SELECT PR, PA, codigo, peso, valor 
    FROM datos 
    WHERE (codigo LIKE '0702%' OR codigo LIKE '0707%' OR codigo LIKE '070960%' 
           OR codigo LIKE '070993%' OR codigo LIKE '070930%')
    AND ano = 23
    """
    df_raw = pd.read_sql(query, engine)

    # 2. IDENTIFICAR EL "TOP"
    # Las 5 provincias que más kilos exportan
    top_provincias = df_raw.groupby('PR')['peso'].sum().nlargest(5).index.tolist()
    
    # Los 15 países que más kilos reciben
    top_paises = df_raw.groupby('PA')['peso'].sum().nlargest(15).index.tolist()

    # 3. FILTRAR EL DATAFRAME
    # Solo nos quedamos con la élite de los datos
    df = df_raw[(df_raw['PR'].isin(top_provincias)) & (df_raw['PA'].isin(top_paises))].copy()
    
    print(f"✅ Analizando el Top 5 PR y Top 15 PA ({len(df)} registros agrupados)")

    # 4. AGRUPACIÓN ANUAL
    df_agrupado = df.groupby(['PR', 'PA', 'codigo']).agg({
        'peso': 'sum',
        'valor': 'sum'
    }).reset_index()
    
    df_agrupado['precio_medio'] = df_agrupado['valor'] / df_agrupado['peso']

    # 5. NORMALIZACIÓN POR MERCADO (Z-Score)
    # Comparamos a las grandes provincias entre sí dentro de cada gran país
    stats = df_agrupado.groupby(['codigo', 'PA'])['precio_medio'].agg(['mean', 'std']).reset_index()
    df_agrupado = df_agrupado.merge(stats, on=['codigo', 'PA'])

    # Z-Score del precio (distancia al promedio del Top)
    df_agrupado['z_score_precio'] = (df_agrupado['precio_medio'] - df_agrupado['mean']) / (df_agrupado['std'] + 1e-6)

    # 6. ISOLATION FOREST
    # Analizamos peso total anual y la desviación de precio
    features = ['peso', 'z_score_precio']
    
    model = IsolationForest(contamination=0.05, random_state=42) # Subimos un poco el % al ser menos datos
    df_agrupado['es_anomalo'] = model.fit_predict(df_agrupado[features])

    # 7. GUARDAR RESULTADOS
    result = df_agrupado[df_agrupado['es_anomalo'] == -1].copy()

    if not result.empty:
        # Añadimos columnas de identificación para saber quiénes son los tops
        result['analisis_tipo'] = "TOP_5PR_15PA"
        result.to_sql('alertas_top_mercado', con=engine, if_exists='replace', index=False)
        print(f"✅ Se han detectado {len(result)} anomalías en los mercados principales.")
        print(f"Provincias analizadas: {top_provincias}")
    else:
        print("No se detectaron anomalías en los mercados top.")

if __name__ == "__main__":
    run_top_market_analysis()
