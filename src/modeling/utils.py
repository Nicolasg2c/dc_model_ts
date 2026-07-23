from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from .config import scoring
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns


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


def compare_datasets_wilcoxon(models, X_mean, y_mean, X_median, y_median, cv, metrics=None, plot_results=True):
    """
    Realiza la prueba de Wilcoxon de rangos con signo para comparar el rendimiento
    de los modelos entrenados con el conjunto de datos de la media (X_mean, y_mean)
    frente al de la mediana (X_median, y_median).

    Args:
        models (dict): Diccionario de modelos {nombre: modelo}.
        X_mean (pd.DataFrame): Variables calculadas con la media.
        y_mean (pd.Series): Target para la media.
        X_median (pd.DataFrame): Variables calculadas con la mediana.
        y_median (pd.Series): Target para la mediana.
        cv: Esquema de validación cruzada (debe ser el mismo para ambos).
        metrics (list, optional): Lista de métricas a comparar.
        plot_results (bool): Si es True, genera gráficos de caja comparativos.

    Returns:
        pd.DataFrame: Tabla con los resultados de la prueba de Wilcoxon.
    """

    if metrics is None:
        metrics = [
            'test_f1_macro',
            'test_balanced_accuracy'
        ]
    else:
        # Asegurar que las métricas tengan el prefijo 'test_'
        metrics = [
            m if m.startswith(('test_', 'train_'))
            else f'test_{m}'
            for m in metrics
        ]

    results = []
    plot_data = []

    for model_name, model in models.items():

        print(f"Evaluando modelo: {model_name}...")

        # Entrenar en ambos datasets usando el mismo esquema de validación cruzada.
        # Se asume que X_mean y X_median contienen los mismos registros,
        # en el mismo orden, y que y_mean e y_median son iguales.
        cv_scores_mean = train_model(
            model,
            X_mean,
            y_mean,
            cv
        )

        cv_scores_median = train_model(
            model,
            X_median,
            y_median,
            cv
        )

        for metric in metrics:

            if (
                metric not in cv_scores_mean
                or metric not in cv_scores_median
            ):
                print(
                    f"Advertencia: la métrica '{metric}' "
                    "no se encontró en los resultados de CV."
                )
                continue

            scores_mean = np.asarray(
                cv_scores_mean[metric]
            )

            scores_median = np.asarray(
                cv_scores_median[metric]
            )

            # Verificar que ambos conjuntos tengan
            # el mismo número de resultados
            if len(scores_mean) != len(scores_median):
                raise ValueError(
                    f"El número de resultados de validación cruzada "
                    f"no coincide para {model_name} - {metric}."
                )

            # -------------------------------------------------
            # Guardar datos para graficar
            # -------------------------------------------------

            metric_label = (
                metric
                .replace("test_", "")
                .replace("_", " ")
                .title()
            )

            for s in scores_mean:

                plot_data.append({
                    "Modelo": model_name,
                    "Métrica": metric_label,
                    "Valor": s,
                    "Dataset": "Media"
                })

            for s in scores_median:

                plot_data.append({
                    "Modelo": model_name,
                    "Métrica": metric_label,
                    "Valor": s,
                    "Dataset": "Mediana"
                })

            # -------------------------------------------------
            # Diferencias pareadas
            # -------------------------------------------------

            diff = (
                scores_mean -
                scores_median
            )

            mean_score_mean = np.mean(
                scores_mean
            )

            mean_score_median = np.mean(
                scores_median
            )

            mean_diff = (
                mean_score_mean -
                mean_score_median
            )

            # Mediana de las diferencias pareadas.
            # Se utiliza para determinar la dirección de la diferencia
            # en concordancia con la naturaleza de la prueba de Wilcoxon.
            median_diff = np.median(
                diff
            )

            # -------------------------------------------------
            # Prueba de Wilcoxon de rangos con signo
            # -------------------------------------------------

            if np.all(diff == 0):

                stat = np.nan
                p_val = 1.0

            else:

                try:

                    stat, p_val = wilcoxon(
                        scores_mean,
                        scores_median,
                        alternative="two-sided"
                    )

                except ValueError as e:

                    print(
                        f"Error en Wilcoxon para "
                        f"{model_name} y {metric}: {e}"
                    )

                    stat = np.nan
                    p_val = np.nan

            # -------------------------------------------------
            # Determinar significancia estadística
            # -------------------------------------------------

            sig = (
                p_val < 0.05
                if not np.isnan(p_val)
                else False
            )

            # -------------------------------------------------
            # Determinar dirección de la diferencia
            # -------------------------------------------------

            if sig:

                if median_diff > 0:

                    conclusion = (
                        "Diferencia significativa a favor de Media"
                    )

                elif median_diff < 0:

                    conclusion = (
                        "Diferencia significativa a favor de Mediana"
                    )

                else:

                    conclusion = (
                        "Diferencia significativa sin dirección clara"
                    )

            else:

                conclusion = (
                    "Sin diferencia estadísticamente significativa"
                )

            # -------------------------------------------------
            # Guardar resultados
            # -------------------------------------------------

            results.append({

                "Modelo":
                    model_name,

                "Métrica":
                    metric_label,

                "Media (Dataset Media)":
                    mean_score_mean,

                "Media (Dataset Mediana)":
                    mean_score_median,

                "Diferencia (Media - Mediana)":
                    mean_diff,

                "Mediana de diferencias":
                    median_diff,

                "Estadístico":
                    stat,

                "p-valor":
                    p_val,

                "Significativo (alpha=0.05)":
                    "Sí" if sig else "No",

                "Conclusión":
                    conclusion
            })

    # ---------------------------------------------------------
    # Crear DataFrame de resultados
    # ---------------------------------------------------------

    df_results = pd.DataFrame(
        results
    )

    print(
        "\nResultados de la comparación entre "
        "Media y Mediana (Prueba de Wilcoxon)"
    )

    print(
        "=" * 80
    )

    # Formatear visualización
    pd.set_option(
        'display.max_columns',
        None
    )

    display(
        df_results.round(4)
    )

    # ---------------------------------------------------------
    # Graficar si se requiere
    # ---------------------------------------------------------

    if plot_results and len(plot_data) > 0:

        df_plot = pd.DataFrame(
            plot_data
        )

        unique_metrics = (
            df_plot["Métrica"].unique()
        )

        n_metrics = len(
            unique_metrics
        )

        # Determinar número de filas y columnas
        n_cols = min(
            2,
            n_metrics
        )

        n_rows = (
            n_metrics +
            n_cols -
            1
        ) // n_cols

        fig, axes_grid = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                7 * n_cols,
                5 * n_rows
            ),
            squeeze=False
        )

        axes_flat = (
            axes_grid.flatten()
        )

        # Colores
        palette = {
            "Media": "#4A90E2",
            "Mediana": "#50E3C2"
        }

        for idx, m_name in enumerate(
            unique_metrics
        ):

            ax = axes_flat[idx]

            df_metric = df_plot[
                df_plot["Métrica"] == m_name
            ]

            sns.boxplot(
                data=df_metric,
                x="Modelo",
                y="Valor",
                hue="Dataset",
                palette=palette,
                ax=ax,
                width=0.5,
                fliersize=3
            )

            ax.set_title(
                f"Distribución de {m_name}",
                fontweight="bold",
                fontsize=12
            )

            ax.set_xlabel(
                "Modelo",
                fontsize=10
            )

            ax.set_ylabel(
                "Métrica Score",
                fontsize=10
            )

            ax.grid(
                axis='y',
                linestyle='--',
                alpha=0.5
            )

            ax.legend(
                title="Dataset"
            )

        # Ocultar subplots vacíos
        for idx in range(
            len(unique_metrics),
            len(axes_flat)
        ):

            fig.delaxes(
                axes_flat[idx]
            )

        plt.tight_layout()
        plt.show()

    return df_results

def graficar_curvas_roc_multiclase(model, model_name, X, y, cv, class_names=['Control', 'DCL', 'Demencia']):
    """
    Genera y grafica las curvas ROC para cada clase usando la estrategia One-vs-Rest (OvR)
    y calcula el score ROC-AUC usando predicciones Out-of-Fold.
    """
    # 1. Crear el pipeline de preprocesamiento idéntico al de tu entrenamiento
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # 2. Binarizar las etiquetas reales (necesario para OvR)
    classes = sorted(np.unique(y))
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    # 3. Obtener probabilidades (o funciones de decisión) Out-of-Fold
    try:
        y_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba')
    except AttributeError:
        decision = cross_val_predict(pipeline, X, y, cv=cv, method='decision_function')
        y_proba = np.exp(decision) / np.sum(np.exp(decision), axis=1, keepdims=True)
        
    # 4. Calcular el ROC-AUC general (Macro)
    macro_roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
    
    # 5. Graficar las curvas ROC para cada clase
    plt.figure(figsize=(8, 6))
    
    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        class_auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        auc_scores[class_names[i]] = class_auc
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC {class_names[i]} (AUC = {class_auc:.4f})')
        
    # Graficar la línea de referencia del clasificador aleatorio
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador Aleatorio (AUC = 0.5000)')
    
    # Estilizado del gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR / 1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR / Sensibilidad)')
    plt.title(f'Curvas ROC Multiclase One-vs-Rest (Macro AUC = {macro_roc_auc:.4f} modelo: {model_name})')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    # Agregar el macro general al diccionario de salida
    auc_scores['Macro_Average'] = macro_roc_auc
    
    return auc_scores


def apply_grid_search(pipeline, parametros, X, y, cv):
    """
    Función para aplicar GridSearchCV a un pipeline de scikit-learn.
    
    Args:
        pipeline: Pipeline de scikit-learn que contiene el modelo y los pasos de preprocesamiento.
        parametros: Diccionario con los parámetros a buscar en el GridSearchCV.
        X: Características predictoras.
        y: Variable objetivo.
        cv: Estrategia de validación cruzada.
    Returns:
        best_model: El mejor modelo encontrado por GridSearchCV.
        best_params: Los mejores parámetros encontrados.
    """

    grid_search = GridSearchCV(pipeline, parametros, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    
    
    return grid_search


    from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.model_selection import GridSearchCV, cross_validate


def nested_cross_validation(
    pipeline,
    parametros,
    X,
    y,
    cv_interno,
    cv_externo,
    scoring
):
    
    # 1. BÚSQUEDA DE HIPERPARÁMETROS EN EL CICLO INTERNO
    # ____________________________________________________________________________
    # Se utiliza GridSearchCV para encontrar la mejor combinación de hiperparámetros
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=cv_interno,
        scoring=scoring,
        n_jobs=-1
    )
    
    # 2. VALIDACIÓN CRUZADA ANIDADA
    # _____________________________________________________________________________
    
    # En cada partición externa:
    # - GridSearchCV busca los mejores hiperparámetros usando únicamente
    #   los datos de entrenamiento de esa partición.
    # - El modelo seleccionado se evalúa sobre el conjunto de prueba externo.
    
    resultados = cross_validate(
        estimator=grid_search,
        X=X,
        y=y,
        cv=cv_externo,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    
    # 3. BÚSQUEDA FINAL DE HIPERPARÁMETROS
    # ____________________________________________________________________________
    
    # Se realiza una nueva búsqueda utilizando todos los datos disponibles.
    # El objetivo es obtener la configuración óptima que se utilizará
    # posteriormente para entrenar el modelo final.
    
    grid_final = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=cv_interno,
        scoring=scoring,
        n_jobs=-1
    )
    
    grid_final.fit(X, y)
    
    
    # 4. OBTENER MEJORES HIPERPARÁMETROS
    # ____________________________________________________________________________

    mejores_parametros = grid_final.best_params_
    mejor_score_interno = grid_final.best_score_
    mejor_modelo = grid_final.best_estimator_
    
    
    # =========================================================================
    # 5. RETORNAR TODOS LOS RESULTADOS
    # =========================================================================
    
    return {
        'test_score': resultados['test_score'],
        'mejores_parametros': mejores_parametros,
        'mejor_score_interno': mejor_score_interno,
        'mejor_modelo': mejor_modelo
    }