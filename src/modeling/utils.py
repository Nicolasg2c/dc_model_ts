from matplotlib import axes
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_validate
import matplotlib.pyplot as plt
from .config import scoring
import numpy as np
import pandas as pd


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


def print_scores(models, X, y, cv_kfold):
    results = []
    results_overfitting = []
    
    for name, model in models.items():
        cv_scores = train_model(model, X, y, cv_kfold)
        
        # medias para calcular la brecha entre entrenamiento y validación/test, para detectar sobreentrenamiento
        train_f1 = cv_scores['train_f1_macro'].mean()
        test_f1 = cv_scores['test_f1_macro'].mean()
        train_balanced_accuracy = cv_scores['train_balanced_accuracy'].mean()
        test_balanced_accuracy = cv_scores['test_balanced_accuracy'].mean()
        train_sens_control = cv_scores['train_sens_control'].mean()
        test_sens_control = cv_scores['test_sens_control'].mean()
        train_sens_dcl = cv_scores['train_sens_dcl'].mean()
        test_sens_dcl = cv_scores['test_sens_dcl'].mean()
        train_sens_demencia = cv_scores['train_sens_demencia'].mean()
        test_sens_demencia = cv_scores['test_sens_demencia'].mean()
        
        # Calcular y almacenar las medias y desviaciones estándar
        results.append({
            "Modelo": name,
            "F1-Score Macro": test_f1,
            "F1-Score (Std)": cv_scores['test_f1_macro'].std(),
            "Balanced Accuracy": cv_scores['test_balanced_accuracy'].mean(),
            "Bal. Acc. (Std)": cv_scores['test_balanced_accuracy'].std(),
            "Sensibilidad Control": cv_scores['test_sens_control'].mean(),
            "Sensibilidad DCL": cv_scores['test_sens_dcl'].mean(),
            "Sensibilidad Demencia": cv_scores['test_sens_demencia'].mean(),
            "Especificidad Control": cv_scores['test_spec_control'].mean(),
            "Especificidad DCL": cv_scores['test_spec_dcl'].mean(),
            "Especificidad Demencia": cv_scores['test_spec_demencia'].mean()
        })
        results_overfitting.append({
            "Modelo": name,
            "F1 (Entrenamiento)": train_f1,
            "F1 (Validación / Test)": test_f1,
            # Si la brecha es muy grande (> 0.15), hay sobreentrenamiento
            "Brecha (Caída)": train_f1 - test_f1,
            "F1 Validación (Std)": cv_scores['test_f1_macro'].std()
        })
        
        
    # Resultados, ordenados de mejor a peor
    df_results = pd.DataFrame(results).sort_values(by="F1-Score Macro", ascending=False)
    print("Resultados de los modelos")
    print("-" * 70)
    display(df_results.round(4))

    # Resultados del overfitting
    df_overfit = pd.DataFrame(results_overfitting).sort_values(by="F1 (Validación / Test)", ascending=False)
    print("Analisis de overfitting")
    print("-" * 85)
    display(df_overfit.round(4))