from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from .config import scoring, seed
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import shap


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
        
        # Extraer los arreglos de métricas por fold
        train_f1_arr = cv_scores['train_f1_macro']
        test_f1_arr = cv_scores['test_f1_macro']
        
        # Calcular la brecha 
        f1_gap_arr = train_f1_arr - test_f1_arr
        
        # Tabla Principal de Resultados
        results.append({
            "Modelo": name,
            "F1-Score Macro": test_f1_arr.mean(),
            "F1-Score (Std)": test_f1_arr.std(),
            "Balanced Accuracy": cv_scores['test_balanced_accuracy'].mean(),
            "Bal. Acc. (Std)": cv_scores['test_balanced_accuracy'].std(),
            "Sensibilidad Control": cv_scores['test_sens_control'].mean(),
            'Sensbilidad Control (Std)': cv_scores['test_sens_control'].std(),
            "Sensibilidad DCL": cv_scores['test_sens_dcl'].mean(),
            'Sensibilidad DCL (Std)': cv_scores['test_sens_dcl'].std(),
            "Sensibilidad Demencia": cv_scores['test_sens_demencia'].mean(),
            'Sensibilidad Demencia (Std)': cv_scores['test_sens_demencia'].std(),
            "Especificidad Control": cv_scores['test_spec_control'].mean(),
            "Especificidad Control (Std)": cv_scores['test_spec_control'].std(),
            "Especificidad DCL": cv_scores['test_spec_dcl'].mean(),
            'Especificidad DCL (Std)': cv_scores['test_spec_dcl'].std(),
            "Especificidad Demencia": cv_scores['test_spec_demencia'].mean(),
            'Especificidad Demencia (Std)': cv_scores['test_spec_demencia'].std()
        })
        
        # Tabla de Overfitting
        results_overfitting.append({
            "Modelo": name,
            "F1 (Entrenamiento)": train_f1_arr.mean(),
            "F1 (Validación / Test)": test_f1_arr.mean(),
            "Brecha (Caída)": f1_gap_arr.mean(),
            "Brecha (Std)": f1_gap_arr.std() 
        })
        
        
    # Mostrar Resultados, ordenados de mejor a peor
    df_results = pd.DataFrame(results).sort_values(by="F1-Score Macro", ascending=False)
    print("Resultados de los modelos")
    print("-" * 70)
    display(df_results.round(4))

    # Mostrar Resultados del overfitting
    df_overfit = pd.DataFrame(results_overfitting).sort_values(by="F1 (Validación / Test)", ascending=False)
    print("Análisis de overfitting")
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
    # Resultados finales
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

    # display(
    #     df_results.round(4)
    # )

    # ---------------------------------------------------------
    # Graficar
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


def nested_cross_validation_multi(
    pipeline,
    parametros,
    X,
    y,
    cv_interno,
    cv_externo,
    scoring,
    metrica_optimizacion='f1_macro'
):
    """
    Ejecuta Nested CV calculando el score de train (validación interna) y test (externo)
    para medir la brecha de rendimiento (overfitting gap).
    """
    
    # 1. BÚSQUEDA DE HIPERPARÁMETROS EN EL CICLO INTERNO
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=cv_interno,
        scoring=metrica_optimizacion,
        n_jobs=-1
    )
    
    # 2. VALIDACIÓN CRUZADA ANIDADA
    resultados_externos = cross_validate(
        estimator=grid_search,
        X=X,
        y=y,
        cv=cv_externo,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True  # <--- Habilitado
    )
    
    # Procesar métricas incluyendo la brecha (Train - Test)
    metricas_resumen = {}
    for nombre_metrica in scoring.keys():
        key_test = f"test_{nombre_metrica}"
        key_train = f"train_{nombre_metrica}"
        
        scores_test = resultados_externos[key_test]
        scores_train = resultados_externos[key_train]
        brechas = scores_train - scores_test  # Diferencia por cada fold externo
        
        metricas_resumen[nombre_metrica] = {
            'test_mean': np.mean(scores_test),
            'test_std': np.std(scores_test),
            'train_mean': np.mean(scores_train),
            'train_std': np.std(scores_train),
            'gap_mean': np.mean(brechas),
            'gap_std': np.std(brechas)
        }
    
    # 3. ENTRENAMIENTO DEL MODELO FINAL
    grid_final = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=cv_interno,
        scoring=metrica_optimizacion,
        n_jobs=-1
    )
    grid_final.fit(X, y)
    
    return {
        'metricas_evaluacion': metricas_resumen,
        'mejores_parametros': grid_final.best_params_,
        'modelo_final': grid_final.best_estimator_
    }


def permutation_importance_cv(
    model,
    X,
    y,
    cv,
    scoring='f1_macro',
    n_repeats=30,
    random_state=seed,
    model_name=None
):
    """
    Calcula la importancia por permutación sobre cada partición
    de validación cruzada y genera una gráfica con el nombre del modelo en el título.
    """
    if model_name is None:
        if isinstance(model, Pipeline):
            model_name = type(model.steps[-1][1]).__name__
        else:
            model_name = type(model).__name__

    importancias_folds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

        y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        modelo_fold = clone(model)
        modelo_fold.fit(X_train, y_train)

        resultado = permutation_importance(
            modelo_fold,
            X_val,
            y_val,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        importancias_folds.append(resultado.importances_mean)

    importancias_folds = np.array(importancias_folds)

    importancia_media = np.mean(importancias_folds, axis=0)
    importancia_std = np.std(importancias_folds, axis=0)

    nombres_variables = [col.replace('_', ' ').capitalize() for col in X.columns]
    
    df_importancia = pd.DataFrame({
        'Variable': nombres_variables,
        'Importancia_Media': importancia_media,
        'Importancia_Std': importancia_std
    }).sort_values(by='Importancia_Media', ascending=False).reset_index(drop=True)

    altura_fig = max(6, len(df_importancia) * 0.4)
    fig, ax = plt.subplots(figsize=(10, altura_fig))

    posiciones = np.arange(len(df_importancia))

    norm = plt.Normalize(
        df_importancia['Importancia_Media'].min(),
        df_importancia['Importancia_Media'].max()
    )
    colores = plt.cm.viridis(norm(df_importancia['Importancia_Media']))

    ax.barh(
        posiciones,
        df_importancia['Importancia_Media'],
        xerr=df_importancia['Importancia_Std'],
        capsize=4,
        color=colores,
        ecolor='black',
        alpha=0.85
    )

    ax.set_yticks(posiciones)
    ax.set_yticklabels(df_importancia['Variable'], fontsize=10)
    ax.invert_yaxis()

    metric_label = scoring.replace('_', ' ').upper() if isinstance(scoring, str) else 'Métrica'
    ax.set_xlabel(f'Disminución promedio en {metric_label}', fontsize=11, labelpad=10)
    ax.set_ylabel('Variable', fontsize=11)
    ax.set_title(f'Importancia de las variables ({model_name})', fontweight='bold', fontsize=13)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    max_reach = (df_importancia['Importancia_Media'] + df_importancia['Importancia_Std']).max()
    min_reach = (df_importancia['Importancia_Media'] - df_importancia['Importancia_Std']).min()

    rango = max_reach - min(0, min_reach)
    offset = rango * 0.02 if rango > 0 else 0.005

    for i, (valor, desviacion) in enumerate(zip(df_importancia['Importancia_Media'], df_importancia['Importancia_Std'])):
        if valor >= 0:
            pos_x = valor + desviacion + offset
            ha_align = 'left'
        else:
            pos_x = valor - desviacion - offset
            ha_align = 'right'

        ax.text(
            pos_x,
            i,
            f'{valor:.4f} ± {desviacion:.4f}',
            va='center',
            ha=ha_align,
            fontsize=8.5,
            fontweight='bold'
        )

    lim_max = max_reach + (rango * 0.25) if max_reach > 0 else 0.05
    lim_min = min_reach - (rango * 0.25) if min_reach < 0 else -0.01
    
    ax.set_xlim(lim_min, lim_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return df_importancia, fig


def shap_importance_models(model, X, model_name="Modelo"):
    """
    Calcula la importancia mediante SHAP evitando el AttributeError
    en las propiedades read-only de Scikit-Learn / Pipeline.
    """
    nombres_cols = [col.replace('_', ' ').capitalize() for col in X.columns]

    print(f"Calculando SHAP para {model_name}...")

    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # Random Forest
    if isinstance(estimator, RandomForestClassifier):
        X_prep = model.steps[0][1].transform(X) if hasattr(model, 'steps') else X
        X_prep_df = pd.DataFrame(X_prep, columns=nombres_cols)
        explainer = shap.TreeExplainer(estimator)
        shap_vals = explainer.shap_values(X_prep_df)

    # Regresión Logística
    elif isinstance(estimator, LogisticRegression):
        X_prep = model.steps[0][1].transform(X) if hasattr(model, 'steps') else X
        X_prep_df = pd.DataFrame(X_prep, columns=nombres_cols)
        explainer = shap.LinearExplainer(estimator, X_prep_df)
        shap_vals = explainer.shap_values(X_prep_df)

    # SVMs
    else:
        if hasattr(model, "decision_function"):
            fn = lambda data: model.decision_function(pd.DataFrame(data, columns=X.columns))
        elif hasattr(model, "predict_proba"):
            fn = lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1]
        else:
            fn = lambda data: model.predict(pd.DataFrame(data, columns=X.columns))

        X_array = X.values if hasattr(X, 'values') else X
        
        idx_sample = np.random.choice(len(X_array), min(30, len(X_array)), replace=False)
        background = X_array[idx_sample]

        explainer = shap.KernelExplainer(fn, background)
        shap_vals = explainer.shap_values(X_array)

    if isinstance(shap_vals, list):
        vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    elif len(np.shape(shap_vals)) == 3:
        vals = shap_vals[:, :, 1]
    else:
        vals = shap_vals

    abs_vals = np.abs(vals)
    mean_shap = np.mean(abs_vals, axis=0)
    std_shap = np.std(abs_vals, axis=0)

    df_shap = pd.DataFrame({
        'Variable': nombres_cols,
        'Importancia_Media': mean_shap,
        'Importancia_Std': std_shap
    }).sort_values(by='Importancia_Media', ascending=False).reset_index(drop=True)
    
    plt.close('all')
    
    altura_fig = max(6, len(df_shap) * 0.45)
    fig, ax = plt.subplots(figsize=(10, altura_fig), dpi=100)

    posiciones = np.arange(len(df_shap))

    min_val = df_shap['Importancia_Media'].min()
    max_val = df_shap['Importancia_Media'].max()
    norm = plt.Normalize(min_val, max_val)
    colores = plt.cm.viridis(norm(df_shap['Importancia_Media']))

    ax.barh(
        posiciones,
        df_shap['Importancia_Media'],
        xerr=df_shap['Importancia_Std'],
        capsize=4,
        color=colores,
        ecolor='black',
        alpha=0.85
    )

    ax.set_yticks(posiciones)
    ax.set_yticklabels(df_shap['Variable'], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Importancia Promedio |SHAP value|', fontsize=11, labelpad=10)
    ax.set_ylabel('Variable', fontsize=11)
    ax.set_title(f'Importancia de las variables mediante SHAP ({model_name})', fontweight='bold', fontsize=13)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    max_reach = (df_shap['Importancia_Media'] + df_shap['Importancia_Std']).max()
    offset = max_reach * 0.02 if max_reach > 0 else 0.005

    for i, (valor, desviacion) in enumerate(zip(df_shap['Importancia_Media'], df_shap['Importancia_Std'])):
        ax.text(
            valor + desviacion + offset,
            i,
            f'{valor:.4f} ± {desviacion:.4f}',
            va='center',
            ha='left',
            fontsize=8.5,
            fontweight='bold'
        )

    limite_derecho = max_reach * 1.35 if max_reach > 0 else 0.1
    ax.set_xlim(-0.001, limite_derecho)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return df_shap, fig


def explicar_clasificacion_individual(model, X, idx_individuo=0, model_name="Modelo", clase_idx=1):
    """
    explicacion individual SHAP compatible con Pipelines, Random Forest, 
    Regresión Logística, SVM (RBF / Lineal) y problemas multiclase (0, 1, 2...).
    
    Todas las explicaciones de modelos probabilísticos se devuelven 
    en la escala directa de PROBABILIDAD (0 a 1).
    """
    nombres_cols = [col.replace('_', ' ').capitalize() for col in X.columns]
    individuo_orig = X.iloc[idx_individuo]

    if hasattr(model, 'steps'):
        estimator = model.steps[-1][1]
        X_prep = model.steps[0][1].transform(X) if len(model.steps) > 1 else X
    else:
        estimator = model
        X_prep = X

    X_prep_df = pd.DataFrame(X_prep, columns=nombres_cols) if isinstance(X_prep, np.ndarray) else X_prep

    print(f"Calculando explicacion individual para individuo #{idx_individuo} ({model_name}) | Clase {clase_idx}...")

    # RANDOM FOREST
    if isinstance(estimator, RandomForestClassifier):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(X_prep_df)
        
        if len(shap_values.shape) == 3:
            exp_individuo = shap_values[idx_individuo, :, clase_idx]
        else:
            exp_individuo = shap_values[idx_individuo]

    # MODELOS CON PROBABILIDAD (Regresión Logística, SVM RBF)
    elif hasattr(model, "predict_proba"):
        def fn_prob(x):
            df_x = pd.DataFrame(x, columns=X.columns)
            return model.predict_proba(df_x)[:, clase_idx]

        background = shap.sample(X, min(50, len(X)), random_state=42)
        explainer = shap.KernelExplainer(fn_prob, background)
        
        sv = explainer.shap_values(X.iloc[[idx_individuo]], nsamples=100)
        
        vals_1d = sv[0] if isinstance(sv, list) else sv
        base_val = explainer.expected_value

        exp_individuo = shap.Explanation(
            values=np.array(vals_1d, dtype=float).ravel(),
            base_values=float(base_val if np.isscalar(base_val) else base_val[0]),
            data=individuo_orig.values,
            feature_names=nombres_cols
        )

    # MODELOS SIN PROBABILIDAD
    else:
        def fn_decision(x):
            df_x = pd.DataFrame(x, columns=X.columns)
            res = model.decision_function(df_x)
            if res.ndim > 1:
                return res[:, clase_idx]
            return res

        background = shap.sample(X, min(50, len(X)), random_state=42)
        explainer = shap.KernelExplainer(fn_decision, background)
        
        sv = explainer.shap_values(X.iloc[[idx_individuo]], nsamples=100)
        vals_1d = sv[0] if isinstance(sv, list) else sv
        base_val = explainer.expected_value

        exp_individuo = shap.Explanation(
            values=np.array(vals_1d, dtype=float).ravel(),
            base_values=float(base_val if np.isscalar(base_val) else base_val[0]),
            data=individuo_orig.values,
            feature_names=nombres_cols
        )

    exp_individuo.data = individuo_orig.values
    exp_individuo.feature_names = nombres_cols

    plt.close('all')
    fig = plt.figure(figsize=(9, 6), dpi=100)
    shap.plots.waterfall(exp_individuo, show=False)
    plt.title(f'explicacion Individual - Individuo #{idx_individuo} ({model_name}) | Clase {clase_idx}', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()