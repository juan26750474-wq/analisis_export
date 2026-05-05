import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

def run_advanced_analysis():
    # 1. CONEXIÓN
    engine = create_engine(f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")

    # Variables de tiempo (Ajusta según tus datos)
    ano_analisis = 23 # El año que queremos vigilar
    ano_inicio = 19   # El año desde el que cogemos el histórico

    # 2. EXTRACCIÓN (Últimos años para crear el histórico)
    query = f"""
    SELECT ano, PR, PA, codigo, peso, valor 
    FROM datos 
    WHERE (codigo LIKE '0702%' OR codigo LIKE '0707%' OR codigo LIKE '070960%' 
           OR codigo LIKE '070993%' OR codigo LIKE '070930%')
    AND ano >= {ano_inicio} AND ano <= {ano_analisis}
    """
    df_raw = pd.read_sql(query, engine)

    # 3. AGRUPACIÓN ANUAL
    df_anual = df_raw.groupby(['ano', 'PR', 'PA', 'codigo']).agg({
        'peso': 'sum',
        'valor': 'sum'
    }).reset_index()

    # 4. CÁLCULO HISTÓRICO (Z-SCORE DE VOLUMEN)
    # Filtramos los años pasados
    historico = df_anual[df_anual['ano'] < ano_analisis].copy()
    stats_historico = historico.groupby(['PR', 'PA', 'codigo'])['peso'].agg(['mean', 'std']).reset_index()
    stats_historico.rename(columns={'mean': 'media_kilos_historico', 'std': 'std_kilos_historico'}, inplace=True)

    # Filtramos el año actual
    df_actual = df_anual[df_anual['ano'] == ano_analisis].copy()
    df_actual['precio_medio'] = df_actual['valor'] / df_actual['peso']

    # Cruzamos el año actual con su propio pasado
    df_final = pd.merge(df_actual, stats_historico, on=['PR', 'PA', 'codigo'], how='left')

    # Calculamos cuánto se desvían los kilos de este año de su propia media histórica
    df_final['std_kilos_historico'] = df_final['std_kilos_historico'].replace(0, np.nan)
    df_final['z_score_peso'] = (df_final['peso'] - df_final['media_kilos_historico']) / df_final['std_kilos_historico']
    df_final['z_score_peso'] = df_final['z_score_peso'].fillna(0) # Si no hay histórico suficiente, lo dejamos neutro

    # 5. FILTRO TOP 5 PROVINCIAS Y TOP 15 PAÍSES
    top_provincias = df_actual.groupby('PR')['peso'].sum().nlargest(5).index.tolist()
    top_paises = df_actual.groupby('PA')['peso'].sum().nlargest(15).index.tolist()
    df_top = df_final[(df_final['PR'].isin(top_provincias)) & (df_final['PA'].isin(top_paises))].copy()

    # 6. CÁLCULO TRANSVERSAL (Z-SCORE DE PRECIO)
    stats_precio = df_top.groupby(['codigo', 'PA'])['precio_medio'].agg(['mean', 'std']).reset_index()
    stats_precio.rename(columns={'mean': 'media_precio_mercado', 'std': 'std_precio_mercado'}, inplace=True)
    
    df_top = pd.merge(df_top, stats_precio, on=['codigo', 'PA'], how='left')
    
    df_top['std_precio_mercado'] = df_top['std_precio_mercado'].replace(0, np.nan)
    df_top['z_score_precio'] = (df_top['precio_medio'] - df_top['media_precio_mercado']) / df_top['std_precio_mercado']
    df_top['z_score_precio'] = df_top['z_score_precio'].fillna(0)

    # 7. MODELO ISOLATION FOREST CON LAS 2 LENTES
    features = ['z_score_peso', 'z_score_precio']
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df_top['es_anomalo'] = model.fit_predict(df_top[features])

    # 8. GUARDAR EN MYSQL
    result = df_top[df_top['es_anomalo'] == -1].copy()

    if not result.empty:
        # Renombramos para que el PHP lo lea fácil
        result.rename(columns={'media_precio_mercado': 'mean'}, inplace=True)
        columnas_guardar = ['PR', 'PA', 'codigo', 'peso', 'valor', 'precio_medio', 'mean', 'z_score_precio', 'media_kilos_historico', 'z_score_peso', 'es_anomalo']
        
        result[columnas_guardar].to_sql('alertas_top_mercado', con=engine, if_exists='replace', index=False)
        print(f"✅ Éxito: {len(result)} anomalías detectadas combinando análisis Histórico y Transversal.")
    else:
        print("No se detectaron anomalías.")

if __name__ == "__main__":
    run_advanced_analysis()
