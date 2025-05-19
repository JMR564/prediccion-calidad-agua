from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Cargar y preparar los datos una sola vez
df = pd.read_csv("datos_calidad_agua.csv")

# Preprocesar datos
df = df.dropna(subset=['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales', 'Nivel'])
X = df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
y = df['Nivel']

# Entrenar el modelo una sola vez
modelo = GradientBoostingClassifier()
modelo.fit(X, y)
niveles = modelo.classes_

@app.route('/')
def formulario():
    return render_template("formulario.html")

@app.route('/resultado', methods=['POST'])
def resultado():
    ciudad = request.form['ciudad']
    datos = df[df['Municipio'] == ciudad]

    if datos.empty:
        return render_template('resultado.html', resultado="Ciudad no encontrada", colores={})

    X_pred = datos[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']].iloc[0:1]
    prediccion_proba = modelo.predict_proba(X_pred)[0]
    prediccion = modelo.predict(X_pred)[0]

    resultado = {niveles[i]: f"{round(prediccion_proba[i]*100, 2)}%" for i in range(len(niveles))}

    colores = {
        'Sin riesgo': 'green',
        'Bajo': 'yellow',
        'Medio': 'orange',
        'Alto': 'red',
        'Inviable sanitariamente': 'darkred'
    }

    return render_template('resultado.html', resultado=resultado, prediccion_final=prediccion, colores=colores)
