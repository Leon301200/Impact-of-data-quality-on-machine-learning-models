# Fonction pour la Complétude

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# Imputation pour les données numériques
def impute_numerical_by_delete(df):
    df_cleaned = df.copy()
    numerical_cols = df_cleaned.select_dtypes(include=np.number).columns
    df_cleaned = df_cleaned.dropna(subset=numerical_cols)
    return df_cleaned

def impute_numerical_by_mean(df):
    df_imputed = df.copy()
    imputer = SimpleImputer(strategy='mean')
    df_imputed[df_imputed.select_dtypes(include='number').columns] = imputer.fit_transform(df_imputed.select_dtypes(include='number'))
    return df_imputed

def impute_numerical_by_median(df):
    df_imputed = df.copy()
    imputer = SimpleImputer(strategy='median')
    df_imputed[df_imputed.select_dtypes(include='number').columns] = imputer.fit_transform(df_imputed.select_dtypes(include='number'))
    return df_imputed

def impute_numerical_by_knn(df):
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=5)
    df_imputed[df_imputed.select_dtypes(include='number').columns] = imputer.fit_transform(df_imputed.select_dtypes(include='number'))
    return df_imputed

def impute_numerical_by_multiple(df):
    df_imputed = df.copy()
    imputer = IterativeImputer()
    df_imputed[df_imputed.select_dtypes(include='number').columns] = imputer.fit_transform(df_imputed.select_dtypes(include='number'))
    return df_imputed

def impute_numerical_by_decision_tree(df):
    df_imputed = df.copy()
    numerical_columns = df_imputed.select_dtypes(include='number').columns
    categorical_columns = df_imputed.select_dtypes(include='object').columns

    for col in numerical_columns:
        if df_imputed[col].isnull().sum() > 0:
            # Séparer les données connues et inconnues
            known_data = df_imputed[df_imputed[col].notna()]
            unknown_data = df_imputed[df_imputed[col].isna()]

            if not unknown_data.empty:
                # Convertir les colonnes catégoriques en dummies
                known_data_dummies = pd.get_dummies(known_data, columns=categorical_columns, drop_first=True)
                unknown_data_dummies = pd.get_dummies(unknown_data, columns=categorical_columns, drop_first=True)

                # Aligner les colonnes pour s'assurer qu'elles correspondent
                unknown_data_dummies = unknown_data_dummies.reindex(columns=known_data_dummies.columns, fill_value=0)

                # Entraîner le modèle sur les données connues
                model = DecisionTreeRegressor()
                model.fit(known_data_dummies.drop(columns=[col]), known_data[col])

                # Prédire les valeurs manquantes et les remplacer
                df_imputed.loc[df_imputed[col].isna(), col] = model.predict(unknown_data_dummies.drop(columns=[col]))

    return df_imputed


# Imputation pour les données catégoriques
def impute_categorical_by_delete(df):
    df_cleaned = df.copy()
    categorical_cols = df_cleaned.select_dtypes(include='object').columns
    df_cleaned = df_cleaned.dropna(subset=categorical_cols)
    return df_cleaned

def impute_categorical_by_mode(df):
    df_imputed = df.copy()
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed[df_imputed.select_dtypes(include='object').columns] = imputer.fit_transform(df_imputed.select_dtypes(include='object'))
    return df_imputed

def impute_categorical_by_new_category(df):
    df_imputed = df.copy()
    df_imputed = df_imputed.fillna('Missing')
    return df_imputed

def impute_categorical_by_hot_deck(df):
    df_imputed = df.copy()
    for col in df_imputed.columns:
        missing_indices = df_imputed[col].isna()
        if missing_indices.any():
            available_values = df_imputed[col].dropna().values
            imputed_values = np.random.choice(available_values, size=missing_indices.sum())
            df_imputed.loc[missing_indices, col] = imputed_values
    return df_imputed

def impute_categorical_by_decision_tree(df):

    df_imputed = df.copy()

    object_columns = df_imputed.select_dtypes(include=['object']).columns

    for col in object_columns:
        if df_imputed[col].isnull().sum() > 0:
            # Diviser le DataFrame en deux ensembles : avec et sans valeurs manquantes pour la colonne actuelle
            df_missing = df_imputed[df_imputed[col].isnull()]
            df_not_missing = df_imputed[~df_imputed[col].isnull()]

            # Si toutes les valeurs sont manquantes, on ne peut rien prédire
            if df_not_missing.empty:
                continue

            # Définir les features (X) et la target (y)
            X = df_not_missing.drop(columns=[col])
            y = df_not_missing[col].astype(str)  # S'assurer que la cible est bien catégorielle

            # Encoder les colonnes de type object restantes
            X = pd.get_dummies(X, drop_first=True)

            # S'assurer que les colonnes de test correspondent bien à celles d'entraînement
            X_missing = pd.get_dummies(df_missing.drop(columns=[col]), drop_first=True).reindex(columns=X.columns, fill_value=0)

            # Créer et entraîner le modèle
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X, y)

            # Prédire les valeurs manquantes
            df_imputed.loc[df_imputed[col].isnull(), col] = model.predict(X_missing)

    return df_imputed

