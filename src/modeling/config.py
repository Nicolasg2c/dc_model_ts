import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, confusion_matrix


#semilla para replicabilidad
seed: int = 19971711


#Funciones para Sensibilidad y Especificidad ---
def get_sensibilidad_clase(y_true, y_pred, clase):
    """ Calcula la sensibilidad (recall) para una clase específica en un problema de clasificación multiclase.
    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas por el modelo.
        clase (int): La clase para la cual se desea calcular la sensibilidad.
        Returns:
            float: La sensibilidad para la clase especificada."""
    # La sensibilidad por clase es equivalente al Recall de esa clase
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return recalls[clase]

def get_especificidad_clase(y_true, y_pred, clase):
    """ Calcula la especificidad para una clase específica en un problema de clasificación multiclase.
    Args:
        y_true (array-like): Etiquetas verdaderas.
        y_pred (array-like): Etiquetas predichas por el modelo.
        clase (int): La clase para la cual se desea calcular la especificidad.
        Returns:
            float: La especificidad para la clase especificada."""
    # Calcula la especificidad usando la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tp = cm[clase, clase]
    fn = np.sum(cm[clase, :]) - tp
    fp = np.sum(cm[:, clase]) - tp
    tn = np.sum(cm) - (tp + fn + fp)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

#Scoring personalizado para la validación cruzada
scoring: dict[str, str | callable] = {
    # Métricas estándar
    'f1_macro': 'f1_macro',
    'balanced_accuracy': 'balanced_accuracy',
    
    # Sensibilidad por clase (Control=0, DCL=1, Demencia=2)
    'sens_control': make_scorer(get_sensibilidad_clase, clase=0),
    'sens_dcl': make_scorer(get_sensibilidad_clase, clase=1),
    'sens_demencia': make_scorer(get_sensibilidad_clase, clase=2),
    
    # Especificidad por clase (Control=0, DCL=1, Demencia=2)
    'spec_control': make_scorer(get_especificidad_clase, clase=0),
    'spec_dcl': make_scorer(get_especificidad_clase, clase=1),
    'spec_demencia': make_scorer(get_especificidad_clase, clase=2)
}

#Validación cruzada
# Configuración de la validación cruzada estratificada repetida / debido a la naturaleza desbalanceada de las clases
cv_repeated_kfold: RepeatedStratifiedKFold = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=seed
) 
#Configuración de la validación cruzada estratificada / debido a la naturaleza desbalanceada de las clases
cv_kfold: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
