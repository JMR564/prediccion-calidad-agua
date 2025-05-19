from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

@app.route('/')
def formulario():
    return render_template("formulario.html")

# Carga inicial del dataset
df = pd.read_csv("datos_calidad_agua.csv")

@app.route('/resultado', methods=['POST'])
def resultado():
    ciudad = request.form['ciudad']
    
    # Filtrar por ciudad
    datos = df[df['Municipio'] == ciudad]

    if datos.empty:
        return render_template('resultado.html', resultado="Ciudad no encontrada", colores={})

    # Variables predictoras y variable objetivo
    X = df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
    y = df['Nivel de riesgo']

    # Entrenar el modelo
    modelo = GradientBoostingClassifier()
    modelo.fit(X, y)

    # Datos de entrada para predicción
    X_pred = datos[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]

    # Predicción
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
