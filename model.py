import torch
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import f1_score, make_scorer




# Modelli di classificazione
class SVMModel:
    def __init__(self):
        self.model = SVC()

    def predict(self, x):
        return self.model.predict(x)
    
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def predict(self, x):
        return self.model.predict(x)





# Funzioni per l'ottimizzazione dei modelli

def optimize_svm_model(X_train, y_train):

    

    new_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC()),
        ]
    )

    param_grid = {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'model__kernel': ['linear', 'poly', 'rbf'],
        'model__gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto'],
        'model__degree': [2, 3, 4],  # Limito i valori per evitare overfitting
        'model__coef0': [0.0, 0.1, 0.5],  # Limito i valori per kernel 'poly' e 'sigmoid'
        'model__class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]  # Diversi pesi per il bilanciamento
    }
    grid_search = GridSearchCV(new_model, param_grid,verbose=5, cv=5, refit=True, scoring="accuracy") # cv=5 indica la divisione in 5 fold per la cross-validation
    grid_search.fit(X_train, y_train) # Addestra il modello con i dati di training

    return grid_search.best_estimator_, grid_search.best_params_ # ritorna il modello ottimizzato e i parametri migliori
    





def optimize_rf_model(X_train, y_train):

    new_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier()),   
        ]
    )

    param_grid = {
        
        'model__n_estimators': [100, 300, 500, 700],
        'model__max_depth': [None, 20, 40, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__bootstrap': [True, False],
        'model__min_samples_leaf': [1, 4],
        'model__max_features': [None, 'sqrt', 'log2'],
        'model__class_weight': [None, 'balanced']
    }


    grid_search = GridSearchCV(new_model, param_grid,verbose=5, cv=5, refit=True, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_