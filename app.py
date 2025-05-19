from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Leer CSV al inicio y preparar datos
df = pd.read_csv("datos_calidad_agua.csv")

# Función para calcular nivel de riesgo
def calcular_nivel(irca):
    try:
        irca = float(irca)
        if irca <= 5:
            return 'Sin riesgo'
        elif irca <= 14:
            return 'Bajo'
        elif irca <= 35:
            return 'Medio'
        elif irca <= 80:
            return 'Alto'
        else:
            return 'Inviable sanitariamente'
    except:
        return None

# Crear columna 'Nivel'
df['Nivel'] = df['IRCA'].apply(calcular_nivel)

# Eliminar filas con valores faltantes en columnas clave
df = df.dropna(subset=['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales', 'Nivel'])

@app.route('/')
def formulario():
    return render_template("formulario.html")

@app.route('/resultado', methods=['POST'])
def resultado():
    ciudad = request.form['ciudad']
    datos = df[df['Municipio'] == ciudad]

    if datos.empty:
        return render_template('resultado.html', resultado="Ciudad no encontrada", colores={})

    # Entrenar modelo con todos los datos
    X = df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
    y = df['Nivel']
    modelo = GradientBoostingClassifier()
    modelo.fit(X, y)

    # Predecir para la ciudad seleccionada
    X_pred = datos[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
    prediccion_proba = modelo.predict_proba(X_pred)[0]
    prediccion = modelo.predict(X_pred)[0]
    niveles = modelo.classes_

    resultado = {niveles[i]: f"{round(prediccion_proba[i]*100, 2)}%" for i in range(len(niveles))}

    colores = {
        'Sin riesgo': 'green',
        'Bajo': 'yellow',
        'Medio': 'orange',
        'Alto': 'red',
        'Inviable sanitariamente': 'darkred'
    }

    return render_template('resultado.html', resultado=resultado, prediccion_final=prediccion, colores=colores)

if __name__ == '__main__':
    app.run(debug=True)
