#semilla para replicabilidad
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, confusion_matrix
import numpy as np


seed: int = 19971711

#metricas de evaluacion
# scoring: list[str] = ['f1_macro', 'balanced_accuracy', 'matthews_corrcoef']

# --- 1. Definición de funciones auxiliares para Sensibilidad y Especificidad ---
def obtener_sensibilidad_clase(y_true, y_pred, clase):
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

def obtener_especificidad_clase(y_true, y_pred, clase):
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

# --- 2. Creación de los Scorers para cross_validate ---

scoring: dict[str, str | callable] = {
    # Métricas estándar
    'f1_macro': 'f1_macro',
    'balanced_accuracy': 'balanced_accuracy',
    
    # Sensibilidad por clase (Control=0, DCL=1, Demencia=2)
    'sens_control': make_scorer(obtener_sensibilidad_clase, clase=0),
    'sens_dcl': make_scorer(obtener_sensibilidad_clase, clase=1),
    'sens_demencia': make_scorer(obtener_sensibilidad_clase, clase=2),
    
    # Especificidad por clase (Control=0, DCL=1, Demencia=2)
    'spec_control': make_scorer(obtener_especificidad_clase, clase=0),
    'spec_dcl': make_scorer(obtener_especificidad_clase, clase=1),
    'spec_demencia': make_scorer(obtener_especificidad_clase, clase=2)
}
