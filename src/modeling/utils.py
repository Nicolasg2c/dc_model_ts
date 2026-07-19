from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from .config import scoring

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