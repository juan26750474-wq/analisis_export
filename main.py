import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

def run_advanced_analysis():
    print("Iniciando análisis de Machine Learning...")

    # 1. CONEXIÓN A LA BASE DE DATOS (Usando los Secrets de GitHub)
    engine = create_engine(f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")

    # 2. EXTRACCIÓN DE DATOS
    # Extraemos todos los datos desde 2019 para tener un buen histórico.
    # IMPORTANTE: Asegúrate de que en tu tabla 'datos' existe la columna 'mes'.
    query = """
    SELECT ano, mes, PR, PA, codigo, peso, valor 
    FROM datos 
    WHERE (codigo LIKE '0702%' OR codigo LIKE '0707%' OR codigo LIKE '070960%' 
           OR codigo LIKE '070993%' OR codigo LIKE '070930%')
    AND ano >= 19
    """
    df_raw = pd.read_sql(query, engine)

    if df_raw.empty:
        print("No se encontraron datos en la base de datos.")
        return

    # 3. CÁLCULO INTELIGENTE DEL AÑO MÓVIL (Últimos 12 meses)
    # Creamos un número combinando Año y Mes para poder ordenarlos y filtrar fácilmente (Ej: 2026 Enero = 2601)
    df_raw['fecha_num'] = df_raw['ano'] * 100 + df_raw['mes']
    
    max_fecha = df_raw['fecha_num'].max()
    limite_actual = max_fecha - 100 # Restamos 100 para retroceder exactamente 1 año (12 meses)
    
    print(f"Último mes detectado en BD: {max_fecha}")
    print(f"Calculando Año Móvil desde {limite_actual + 1} hasta {max_fecha}...")

    # Separamos el Año Móvil actual de los datos Históricos
    df_actual = df_raw[df_raw['fecha_num'] > limite_actual].copy()
    df_historico = df_raw[df_raw['fecha_num'] <= limite_actual].copy()

    # 4. AGRUPACIÓN Y ESTADÍSTICAS HISTÓRICAS
    # A) Comprimimos todo el Año Móvil en un solo total por ruta
    df_actual_agrupado = df_actual.groupby(['PR', 'PA', 'codigo']).agg({
        'peso': 'sum',
        'valor': 'sum'
    }).reset_index()
    df_actual_agrupado['precio_medio'] = df_actual_agrupado['valor'] / df_actual_agrupado['peso']

    # B) Agrupamos el histórico por AÑO natural para sacar la normalidad de años pasados
    df_historico_anual = df_historico.groupby(['ano', 'PR', 'PA', 'codigo']).agg({
        'peso': 'sum'
    }).reset_index()
    
    # Calculamos la Media y la Desviación Estándar de los kilos exportados en el pasado
    stats_historico = df_historico_anual.groupby(['PR', 'PA', 'codigo'])['peso'].agg(['mean', 'std']).reset_index()
    stats_historico.rename(columns={'mean': 'media_kilos_historico', 'std': 'std_kilos_historico'}, inplace=True)

    # Cruzamos el Año Móvil actual con su propio pasado
    df_final = pd.merge(df_actual_agrupado, stats_historico, on=['PR', 'PA', 'codigo'], how='left')

    # 5. FILTROS DE RELEVANCIA (Para limpiar la "basura" estadística)
    MIN_KILOS_ANUALES = 50000        # Ignorar rutas que mueven menos de 50.000 kg al año
    MIN_PORCENTAJE_HISTORICO = 0.15  # Ignorar si exportaron menos del 15% de su histórico habitual

    df_final = df_final[df_final['peso'] >= MIN_KILOS_ANUALES]
    df_final = df_final[
        (df_final['media_kilos_historico'].isna()) | 
        (df_final['peso'] >= (df_final['media_kilos_historico'] * MIN_PORCENTAJE_HISTORICO))
    ]

    # Cálculo de Z-SCORE de VOLUMEN (Análisis Histórico)
    df_final['std_kilos_historico'] = df_final['std_kilos_historico'].replace(0, np.nan)
    df_final['z_score_peso'] = (df_final['peso'] - df_final['media_kilos_historico']) / df_final['std_kilos_historico']
    df_final['z_score_peso'] = df_final['z_score_peso'].fillna(0) # Neutro si no hay histórico

    # 6. FILTRO DE COMPETITIVIDAD (Top 5 Provincias y Top 15 Países)
    top_provincias = df_actual_agrupado.groupby('PR')['peso'].sum().nlargest(5).index.tolist()
    top_paises = df_actual_agrupado.groupby('PA')['peso'].sum().nlargest(15).index.tolist()
    df_top = df_final[(df_final['PR'].isin(top_provincias)) & (df_final['PA'].isin(top_paises))].copy()

    # 7. CÁLCULO DE Z-SCORE DE PRECIO (Análisis Transversal vs Competencia)
    if not df_top.empty:
        # Qué precio medio tiene el mercado en el Año Móvil para ese país y producto
        stats_precio = df_top.groupby(['codigo', 'PA'])['precio_medio'].agg(['mean', 'std']).reset_index()
        # Llamamos 'mean' a la media porque el PHP la espera con ese nombre
        stats_precio.rename(columns={'mean': 'mean', 'std': 'std_precio_mercado'}, inplace=True) 
        
        df_top = pd.merge(df_top, stats_precio, on=['codigo', 'PA'], how='left')
        
        df_top['std_precio_mercado'] = df_top['std_precio_mercado'].replace(0, np.nan)
        df_top['z_score_precio'] = (df_top['precio_medio'] - df_top['mean']) / df_top['std_precio_mercado']
        df_top['z_score_precio'] = df_top['z_score_precio'].fillna(0)

        # 8. MODELO DE MACHINE LEARNING (ISOLATION FOREST)
        # Usamos las dos lentes para buscar al "Bicho Raro"
        features = ['z_score_peso', 'z_score_precio']
        model = IsolationForest(contamination=0.05, random_state=42)
        df_top['es_anomalo'] = model.fit_predict(df_top[features])

        # 9. GUARDAR RESULTADOS EN LA BASE DE DATOS MYSQL
        # Filtramos solo los resultados anómalos (-1)
        result = df_top[df_top['es_anomalo'] == -1].copy()

        if not result.empty:
            # Seleccionamos y ordenamos las columnas tal cual las espera el PHP
            columnas_guardar = ['PR', 'PA', 'codigo', 'peso', 'valor', 'precio_medio', 'mean', 'z_score_precio', 'media_kilos_historico', 'z_score_peso', 'es_anomalo']
            
            # Borramos la tabla vieja y subimos la nueva alerta
            result[columnas_guardar].to_sql('alertas_top_mercado', con=engine, if_exists='replace', index=False)
            print(f"✅ Éxito: {len(result)} anomalías RELEVANTES detectadas y guardadas.")
        else:
            # Si no hay anomalías, guardamos una tabla vacía para que el PHP muestre "Todo en orden"
            pd.DataFrame(columns=['PR', 'PA', 'codigo', 'peso', 'valor', 'precio_medio', 'mean', 'z_score_precio', 'media_kilos_historico', 'z_score_peso', 'es_anomalo']).to_sql('alertas_top_mercado', con=engine, if_exists='replace', index=False)
            print("No se detectaron anomalías relevantes tras aplicar los filtros de significancia.")
    else:
        print("No hay datos que superen los umbrales de relevancia.")

if __name__ == "__main__":
    run_advanced_analysis()
