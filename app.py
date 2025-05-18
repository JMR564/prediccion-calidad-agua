from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


app = Flask(__name__)

@app.route('/')
def formulario():
    return render_template("formulario.html")

df = pd.read_csv("datos_calidad_agua.csv")

@app.route('/resultado', methods=['POST'])
def resultado():
    ciudad = request.form['ciudad']
    
    # Leer el CSV
    df = pd.read_csv("datos_calidad_agua.csv")

    # Filtrar por ciudad
    datos = df[df['Municipio'] == ciudad]

    if datos.empty:
        return render_template('resultado.html', resultado="Ciudad no encontrada", colores={})

    # Separar variables de entrada y salida
    X = df[['pH', 'Turbiedad', 'Color', 'Coliformes']]
    y = df['Nivel']

    # Entrenar modelo (Gradient Boosting por ejemplo)
    from sklearn.ensemble import GradientBoostingClassifier
    modelo = GradientBoostingClassifier()
    modelo.fit(X, y)

    # Datos de entrada para predicción
    X_pred = datos[['pH', 'Turbiedad', 'Color', 'Coliformes']]

    # Realizar predicción
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

