# Fonction pour la Complétude

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# Imputation pour les données numériques
def impute_numerical_mean(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include='number').columns] = imputer.fit_transform(df.select_dtypes(include='number'))
    return df

def impute_numerical_median(df):
    imputer = SimpleImputer(strategy='median')
    df[df.select_dtypes(include='number').columns] = imputer.fit_transform(df.select_dtypes(include='number'))
    return df

def impute_numerical_knn(df):
    imputer = KNNImputer(n_neighbors=5)
    df[df.select_dtypes(include='number').columns] = imputer.fit_transform(df.select_dtypes(include='number'))
    return df

def impute_numerical_multiple(df):
    imputer = IterativeImputer()
    df[df.select_dtypes(include='number').columns] = imputer.fit_transform(df.select_dtypes(include='number'))
    return df

def impute_numerical_decision_tree(df):
    numerical_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(include='object').columns

    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            # Séparer les données connues et inconnues
            known_data = df[df[col].notna()]
            unknown_data = df[df[col].isna()]

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
                df.loc[df[col].isna(), col] = model.predict(unknown_data_dummies.drop(columns=[col]))

    return df


# Imputation pour les données catégoriques
def impute_categorical_mode(df):
    imputer = SimpleImputer(strategy='most_frequent')
    df[df.select_dtypes(include='object').columns] = imputer.fit_transform(df.select_dtypes(include='object'))
    return df

def impute_categorical_new_category(df):
    df = df.fillna('Missing')
    return df

def impute_categorical_decision_tree(df):
    # Créer une copie du dataframe pour ne pas modifier l'original
    df_imputed = df.copy()

    # Sélectionner les colonnes de type 'object'
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

