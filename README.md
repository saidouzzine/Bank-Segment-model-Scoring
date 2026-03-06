# Bank-Segment-model-Scoring
Modélisation prédictive pour identifier les clients les plus susceptibles d’évoluer vers un segment bancaire plus favorable.
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2.2-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-green.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.2-yellow.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

---

# 1. Présentation du projet

Ce projet vise à construire un score prédictif permettant d’identifier les clients ayant le plus fort potentiel d’évolution vers un segment bancaire plus favorable. L’objectif est de sélectionner les 1000 clients les plus susceptibles de changer de segment afin d’orienter les actions commerciales.

---

# 2. Objectifs

1. Décrire et analyser la base clients.  
2. Nettoyer et structurer les données.  
3. Créer des variables explicatives pertinentes.  
4. Tester plusieurs modèles supervisés.  
5. Comparer les performances et sélectionner le meilleur modèle.  
6. Produire un score final pour les données test.

---

# 3. Données utilisées

La base contient 70 156 clients et 20 variables :  
- Variables numériques : âge, ancienneté, surface financière, mouvements, paiements, épargne…  
- Variables binaires : détention de produits (compte courant, carte, épargne, assurance, crédits).  
- Variable cible : `flag_changement_segment` (0/1).

---

# 4. Préparation et nettoyage des données

1. Traitement des valeurs manquantes (médiane, imputations conditionnelles).  
2. Correction des incohérences (valeurs négatives, distributions anormales).  
3. Remplissage logique pour les clients ne possédant pas certains produits.  
4. Vérification de la cohérence des variables financières.  
5. Analyse de la multicolinéarité et des corrélations.

---

# 5. Catégorisation et Feature Engineering

### 5.1 Catégorisation
- `age_category` : 25–40, 41–60, 61–80, 81–120  
- `anciennete_category` : 0–10, 11–20, 21–30, 31–120  

### 5.2 Nouvelles variables créées
- ratio_solde_moyen  
- flag_actif_compte_courant  
- nb_produits_diversifies  
- score_engagement  
- ratio_encours_total_moyen  
- ratio_epargne_investie_par_liquide  

Ces variables enrichissent la compréhension du comportement financier des clients.

---

# 6. Échantillonnage

Deux approches ont été utilisées :  
1. **Échantillonnage classique** : 80% apprentissage / 20% validation.  
2. **Échantillonnage équilibré** : sous-échantillonnage de la classe majoritaire pour compenser le déséquilibre (90% / 10%).

Les distributions ont été vérifiées pour garantir la représentativité.

---

# 7. Modélisation

Les modèles suivants ont été testés :  
- Régression logistique  
- Random Forest  
- LightGBM  
- HistGradientBoostingClassifier  
- CatBoostClassifier  

Chaque modèle a été évalué sur les deux types d’échantillonnage.

---

# 8. Évaluation et comparaison

Métriques utilisées :  
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Balanced Accuracy  
- AUC ROC  
- Matrices de confusion  

Les modèles boosting (LightGBM, HGB, CatBoost) offrent les meilleures performances, notamment sur la classe minoritaire.

---

# 9. Prédiction sur les données test

1. Sélection du meilleur modèle selon AUC et Recall.  
2. Analyse de l’importance des variables.  
3. Génération du score final pour chaque client.  
4. Classement des 1000 clients les plus susceptibles de changer de segment.

---

# 10. Structure du dépôt

```
project/
├── data/                     # Données brutes (non incluses)
├── notebooks/                # Analyses exploratoires
├── models/                   # Modèles entraînés
├── figures/                  # Visualisations
├── scoring_bank.py           # Script principal
├── requirements.txt
└── README.md
```

---

# 11. Technologies utilisées

- Python  
- Pandas, NumPy  
- Scikit-learn  
- LightGBM  
- CatBoost  
- Matplotlib, Seaborn  

---

# 12. Auteur

Projet réalisé par Said Ouzzine  
https://www.linkedin.com/in/said-ouzzine/


