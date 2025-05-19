import pandas as pd

def obtener_datos_por_ciudades(ciudades):
    try:
        # Lee el archivo CSV que debe estar en la raíz del proyecto
        df = pd.read_csv("datos_calidad_agua.csv")

        # Filtra las filas que correspondan a las ciudades indicadas
        df_ciudades = df[df["Municipio"].isin(ciudades)]

        return df_ciudades
    except Exception as e:
        print("Error al leer o filtrar el CSV:", e)
        return pd.DataFrame()
