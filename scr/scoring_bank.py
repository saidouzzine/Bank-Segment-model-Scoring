# ============================================================
# Bank Segment model Scoring – Data Science Pipeline
# Auteur : Said Ouzzine
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve, auc
)

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib


# ============================================================
# 1. Chargement des données
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    return df


# ============================================================
# 2. Nettoyage des données
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Imputation conditionnelle
    median_m6 = df.loc[df['anciennete'] < 1, 'surface_financiere_totale_M6'].median()
    df['surface_financiere_totale_M6'].fillna(median_m6, inplace=True)

    # Compte courant
    if (df['flag_compte_courant'] == 0).any():
        df[['encours_compte_courant',
            'nb_mouvements_compte_courant',
            'mt_mouvements_compte_courant']] = df[
                ['encours_compte_courant',
                 'nb_mouvements_compte_courant',
                 'mt_mouvements_compte_courant']
            ].fillna(0)

    # Carte de paiement
    if (df['flag_carte_paiement'] == 0).any():
        df[['nb_paiements_carte', 'mt_paiements_carte']] = df[
            ['nb_paiements_carte', 'mt_paiements_carte']
        ].fillna(0)

    # Épargne liquide
    if (df['flag_epargne_liquide'] == 0).any():
        df['encours_epargne_liquide'].fillna(0, inplace=True)

    return df


# ============================================================
# 3. Feature Engineering
# ============================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:

    # Catégorisation
    df['age_category'] = pd.cut(df['age'], bins=[25, 40, 60, 80, 120],
                                labels=['25-40', '41-60', '61-80', '81-120'])

    df['anciennete_category'] = pd.cut(df['anciennete'], bins=[0, 10, 20, 30, 120],
                                       labels=['0-10', '11-20', '21-30', '31-120'])

    # Nouvelles variables
    df['ratio_solde_moyen'] = df['encours_compte_courant'] / df['surface_financiere_totale']
    df['flag_actif_compte_courant'] = (
        df['nb_mouvements_compte_courant'] >
        df['nb_mouvements_compte_courant'].mean()
    ).astype(int)

    df['nb_produits_diversifies'] = df[
        ['flag_compte_courant', 'flag_carte_paiement', 'flag_epargne_liquide',
         'flag_epargne_investie', 'flag_assurance', 'flag_credit_consommation',
         'flag_credit_immobilier']
    ].sum(axis=1)

    df['score_engagement'] = (
        df['encours_compte_courant'] +
        df['encours_epargne_liquide'] +
        df['encours_epargne_investie']
    )

    avg_encours = df['surface_financiere_totale'].mean()
    df['ratio_encours_total_moyen'] = df['surface_financiere_totale'] / avg_encours

    df['ratio_epargne_investie_par_liquide'] = (
        df['encours_epargne_investie'] /
        (df['encours_epargne_liquide'] + 1)
    )

    # Imputations finales
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['age_category'].fillna(df['age_category'].mode()[0], inplace=True)
    df['anciennete_category'].fillna(df['anciennete_category'].mode()[0], inplace=True)

    return df


# ============================================================
# 4. Encodage
# ============================================================

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['id_client'], errors='ignore')
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


# ============================================================
# 5. Modélisation
# ============================================================

def get_models():
    return {
        'GradientBoosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [300, 600],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 4, 5]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [200, 400],
                'max_depth': [None, 10, 20]
            }
        },
        'HistGradientBoosting': {
            'model': HistGradientBoostingClassifier(),
            'params': {
                'max_iter': [600, 800],
                'max_leaf_nodes': [300, 500],
                'class_weight': ['balanced']
            }
        },
        'LGBMClassifier': {
            'model': LGBMClassifier(),
            'params': {
                'n_estimators': [100, 150],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 4, 5]
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(),
            'params': {
                'n_estimators': [300],
                'learning_rate': [0.1],
                'max_depth': [5]
            }
        }
    }


# ============================================================
# 6. Entraînement + Sélection du meilleur modèle
# ============================================================

def train_and_select(X_train, y_train, X_val, y_val):
    models = get_models()
    best_model = None
    best_auc = 0

    for name, cfg in models.items():
        print(f"\nTraining {name}...")

        search = RandomizedSearchCV(
            cfg['model'],
            cfg['params'],
            n_iter=10,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)
        auc_val = roc_auc_score(y_val, search.predict_proba(X_val)[:, 1])

        print(f"AUC Validation: {auc_val:.4f}")

        if auc_val > best_auc:
            best_auc = auc_val
            best_model = search

    return best_model


# ============================================================
# 7. Main Pipeline
# ============================================================

def main():
    df = load_data("Base_historique.txt")
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode(df)

    X = df.drop('flag_changement_segment', axis=1)
    y = df['flag_changement_segment']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model = train_and_select(X_train, y_train, X_val, y_val)

    print("\nBest model selected.")
    joblib.dump(best_model, "best_model.joblib")


if __name__ == "__main__":
    main()
