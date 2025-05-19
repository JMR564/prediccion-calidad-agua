from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Leer y preparar los datos
df = pd.read_csv("datos_calidad_agua.csv")

# Crear la columna "Nivel" si no existe
if 'Nivel' not in df.columns:
    def clasificar_irca(irca):
        if irca <= 5:
            return 'Sin riesgo'
        elif irca <= 14:
            return 'Riesgo bajo'
        elif irca <= 35:
            return 'Riesgo medio'
        elif irca <= 80:
            return 'Riesgo alto'
        else:
            return 'Riesgo inviable sanitariamente'
    df['Nivel'] = df['IRCA'].apply(clasificar_irca)

# Entrenar el modelo una sola vez
X = df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
y = df['Nivel']
modelo = GradientBoostingClassifier()
modelo.fit(X, y)

@app.route('/')
def formulario():
    return render_template("formulario.html")

@app.route('/resultado', methods=['POST'])
def resultado():
    ciudad = request.form['ciudad']
    datos = df[df['Municipio'] == ciudad]

    if datos.empty:
        return render_template('resultado.html', resultado="Ciudad no encontrada", colores={})

    X_pred = datos[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
    prediccion_proba = modelo.predict_proba(X_pred)[0]
    prediccion = modelo.predict(X_pred)[0]

    niveles = modelo.classes_
    resultado_dict = {niveles[i]: round(prediccion_proba[i]*100, 2) for i in range(len(niveles))}

    colores = {
        'Sin riesgo': 'green',
        'Riesgo bajo': 'yellow',
        'Riesgo medio': 'orange',
        'Riesgo alto': 'red',
        'Riesgo inviable sanitariamente': 'darkred'
    }

    return render_template('resultado.html', resultado=resultado_dict, prediccion_final=prediccion, colores=colores)
