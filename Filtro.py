import mysql.connector
import pandas as pd

def obtener_datos_por_ciudades(ciudades):
    try:
        conexion = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="calidad_agua_irca"
        )
        cursor = conexion.cursor(dictionary=True)

        if ciudades:
            placeholders = ', '.join(['%s'] * len(ciudades))
            consulta = f"""
                SELECT *
                FROM calidad_del_agua_para_consumo_humano_en_colombia_20250514
                WHERE Municipio IN ({placeholders})
            """
            cursor.execute(consulta, ciudades)
        else:
            consulta = "SELECT * FROM calidad_del_agua_para_consumo_humano_en_colombia_20250514"
            cursor.execute(consulta)

        resultados = cursor.fetchall()
        conexion.close()
        return pd.DataFrame(resultados)

    except mysql.connector.Error as error:
        print("Error al conectar o consultar:", error)
        return pd.DataFrame()
