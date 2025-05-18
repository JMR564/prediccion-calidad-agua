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
    try:
        ph = float(request.form.get("ph"))
        turbiedad = float(request.form.get("turbiedad"))
        color = float(request.form.get("color"))
        coliformes_totales = float(request.form.get("coliformes_totales"))
        coliformes_fecales = float(request.form.get("coliformes_fecales"))
    except (ValueError, TypeError):
        return render_template("resultado.html", mensaje="Error: todos los valores deben ser numéricos y estar completos.")

    ciudad = request.form.get("ciudad")
    datos = obtener_datos_por_ciudades([ciudad])
    if datos.empty:
        return render_template("resultado.html", mensaje=f"No se encontraron datos para la ciudad: {ciudad}")

    df = datos.replace("ND", pd.NA).dropna()
    if df.empty:
        return render_template("resultado.html", mensaje="No hay datos válidos después de limpiar los 'ND'.")

    try:
        df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']] = df[[
            'pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']].astype(float)
    except KeyError:
        return render_template("resultado.html", mensaje="Faltan columnas esperadas en los datos.")

    def clasificar_irca(irca):
        if irca <= 5:
            return "Sin riesgo"
        elif irca <= 14:
            return "Riesgo bajo"
        elif irca <= 35:
            return "Riesgo medio"
        elif irca <= 80:
            return "Riesgo alto"
        else:
            return "Riesgo inviable sanitariamente"

    if 'Nivel de riesgo' not in df.columns:
        df['Nivel de riesgo'] = df['IRCA'].astype(float).apply(clasificar_irca)

    X = df[['pH', 'Turbiedad', 'Color', 'Coliformes_Totales', 'Coliformes_Fecales']]
    y = df['Nivel de riesgo']

    if len(y.unique()) < 2:
        return render_template("resultado.html", mensaje="No hay suficientes clases para entrenar el modelo.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = GradientBoostingClassifier()
    modelo.fit(X_train, y_train)

    entrada = pd.DataFrame([{
        "pH": ph,
        "Turbiedad": turbiedad,
        "Color": color,
        "Coliformes_Totales": coliformes_totales,
        "Coliformes_Fecales": coliformes_fecales
    }])

    probabilidades = modelo.predict_proba(entrada)[0]
    clases = modelo.classes_
    predicciones = dict(zip(clases, [round(p * 100, 2) for p in probabilidades]))
    clase_predicha = clases[np.argmax(probabilidades)]

    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    ruta_img = os.path.join("static", "matriz_confusion.png")
    plt.savefig(ruta_img)
    plt.close()

    return render_template("resultado.html", clase_predicha=clase_predicha, predicciones=predicciones, imagen=ruta_img)

if __name__ == "__main__":
    from os import environ
    port = int(environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
