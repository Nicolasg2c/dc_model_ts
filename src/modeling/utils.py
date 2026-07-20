from matplotlib import axes
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_validate
import matplotlib.pyplot as plt
from .config import scoring
import numpy as np


def train_model(model, X, y, cv):
     # Se crea un pipeline que incluye imputación, aunque el conjunto de datos ya esta preprocesado, es una buena práctica para evitar problemas con NaNs en otros conjuntos de datos.
    # 1: Rellenar NaNs con la mediana (SimpleImputer)
    # 2: Escalar los datos a Media=0, Std=1 (StandardScaler)
    # 3: Entrenar el clasificador
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)

    return cv_scores


def get_confusion_matrix(models, X, y, cv_kfold): 
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    for ax, (model_name, model) in zip(axes, models.items()):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
         #Se generan de nuevo las predicciones para todo el dataset, pero cada paciente es evaluado solo cuando su fold fue el "fold de prueba".
        y_pred = cross_val_predict(pipeline, X, y, cv=cv_kfold)
        
        # Calculo de la matriz de confusión y visualización
        cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "DCL", "Demencia"])

        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=14)
        ax.set_xlabel("Predicción del Modelo", fontsize=11)
        ax.set_ylabel("Diagnóstico Clínico Real", fontsize=11)

    plt.tight_layout()
    plt.show()