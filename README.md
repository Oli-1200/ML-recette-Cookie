# Projet IA de Recette de Cookies

## Description du Projet

Ce projet vise à créer la meilleure recette de cookies en utilisant des techniques d'apprentissage automatique, en particulier TensorFlow. Inspiré par le billet de blog de Sara Robinson sur la classification des recettes de pain, de gâteau et de cookies à l'aide de l'IA, ce projet va plus loin en tentant de générer une recette de cookies optimale.

Pour en savoir plus sur la genèse du projet : https://olivierleclere.ch/blog/2024/09/jeune-genevois-2-0-quand-l-ia-patisse/ 

## Structure du Projet

Le projet se compose de deux scripts Python principaux :

1. `train-cookie-recipe.py` : Ce script est responsable de l'entraînement du modèle d'apprentissage automatique à l'aide de TensorFlow. Il traite les données d'entrée, crée et entraîne le modèle, puis sauvegarde le modèle entraîné et les artefacts nécessaires pour une utilisation ultérieure.

2. `generate-best-recipe.py` : Ce script utilise le modèle entraîné pour générer la meilleure recette de cookies possible basée sur les paramètres appris.

## Prérequis

Pour exécuter le script `train-cookie-recipe.py`, vous avez besoin de deux fichiers supplémentaires :

1. `recettes_cookies.csv` : Un fichier CSV contenant l'ensemble des données des recettes de cookies avec leurs ingrédients, paramètres de cuisson et notes.
2. `ingredient_columns.json` : Un fichier JSON qui définit la structure des données des ingrédients, séparant les ingrédients principaux, les ingrédients mineurs et les paramètres de cuisson.

## Installation et Utilisation

1. Assurez-vous d'avoir Python installé sur votre système.
2. Installez les dépendances requises :
   ```
   pip install tensorflow pandas numpy scikit-learn joblib matplotlib
   ```
3. Placez les fichiers `recettes_cookies.csv` et `ingredient_columns.json` dans le même répertoire que les scripts Python.
4. Exécutez le script d'entraînement :
   ```
   python train-cookie-recipe.py
   ```
5. Une fois l'entraînement terminé, exécutez le script de génération :
   ```
   python generate-best-recipe.py
   ```

## Fonctionnement

### Entraînement (`train-cookie-recipe.py`)

1. Charge et prétraite les données des recettes à partir de `recettes_cookies.csv`.
2. Définit la structure du modèle de réseau neuronal en utilisant TensorFlow.
3. Entraîne le modèle sur les données prétraitées.
4. Sauvegarde le modèle entraîné, les scalers et autres artefacts nécessaires pour une utilisation ultérieure.

### Génération de Recette (`generate-best-recipe.py`)

1. Charge le modèle entraîné et les artefacts nécessaires.
2. Utilise des techniques d'optimisation pour trouver la meilleure combinaison d'ingrédients et de paramètres de cuisson.
3. Génère une recette complète de cookies basée sur les paramètres optimisés.
4. Affiche la recette en grammes et en ratios, ainsi que les instructions de cuisson.

## Sortie

Le script `generate-best-recipe.py` affichera la meilleure recette de cookies, comprenant :

- Les quantités d'ingrédients en grammes
- La température et la durée de cuisson
- L'ordre d'incorporation des ingrédients
- La note prédite pour la recette

## Contribution

Les contributions pour améliorer le modèle ou étendre le projet sont les bienvenues. N'hésitez pas à soumettre des pull requests ou à ouvrir des issues pour en discuter.

## Licence

GNU AGPLv3

## Remerciements

- Sara Robinson pour l'inspiration de ce projet
- L'équipe TensorFlow pour leur excellente bibliothèque d'apprentissage automatique
