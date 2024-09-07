import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import joblib
import json
from scipy.optimize import minimize
import os
from datetime import datetime

@tf.keras.utils.register_keras_serializable()
def log_transform(x):
    return tf.math.log1p(x)

# Check if required files are available
required_files = ['cookie_model.keras', 'ingredient_scaler.joblib', 'cooking_scaler.joblib', 'ingredient_columns.json', 'columns_bounds.json', 'order_mapping.json', 'orders.json']
for file in required_files:
    if not os.path.exists(file):
        print(f"Error: The file {file} does not exist. Ensure that the training script has been executed.")
        exit(1)

# Load the model and scalers
model = keras.models.load_model('cookie_model.keras', custom_objects={'log_transform': log_transform})
ingredient_scaler = joblib.load('ingredient_scaler.joblib')
cooking_scaler = joblib.load('cooking_scaler.joblib')

# Load ingredient columns and bounds
with open('ingredient_columns.json', 'r') as f:
    ingredient_data = json.load(f)
major_ingredient_columns = ingredient_data["major_ingredient_columns"]
minor_ingredient_columns = ingredient_data["minor_ingredient_columns"]
cooking_parameter_columns = ingredient_data["cooking_parameter_columns"]

with open('columns_bounds.json', 'r') as f:
    columns_bounds = json.load(f)

# Load order mapping
with open('order_mapping.json', 'r') as f:
    order_to_int = json.load(f)

# Invert the order_to_int mapping
int_to_order = {v: k for k, v in order_to_int.items()}

# Load orders
with open('orders.json', 'r') as f:
    orders_data = json.load(f)
    orders = orders_data['orders']

# Extract min and max score from columns_bounds
min_score, max_score = columns_bounds['score']

# Assign bounds to major ingredients, minor ingredients, and cooking parameters
major_bounds = [(columns_bounds[col][0], columns_bounds[col][1]) for col in major_ingredient_columns]
minor_bounds = [(columns_bounds[col][0], columns_bounds[col][1]) for col in minor_ingredient_columns]
cooking_bounds = [(columns_bounds[col][0], columns_bounds[col][1]) for col in cooking_parameter_columns]

# Combine the bounds
bounds = major_bounds + minor_bounds + cooking_bounds

# Prediction function for the score
def predict_score(params):
    # Split major ingredients, minor ingredients, and cooking parameters
    major_params = params[:len(major_ingredient_columns)]
    minor_params = params[len(major_ingredient_columns):len(major_ingredient_columns)+len(minor_ingredient_columns)]
    cooking_params = params[len(major_ingredient_columns)+len(minor_ingredient_columns):]
    
    # Prepare input data as DataFrames
    input_major = pd.DataFrame([major_params], columns=major_ingredient_columns)
    input_minor = pd.DataFrame([minor_params], columns=minor_ingredient_columns)
    input_cooking = pd.DataFrame([cooking_params], columns=cooking_parameter_columns)
    
    # Scale major ingredients and cooking parameters
    input_major_scaled = pd.DataFrame(ingredient_scaler.transform(input_major), columns=major_ingredient_columns)
    input_cooking_scaled = pd.DataFrame(cooking_scaler.transform(input_cooking), columns=cooking_parameter_columns)
    
    # Log transform minor ingredients
    input_minor_log = np.log1p(input_minor)
    
    # Predict the score using the model
    score = model.predict([input_major_scaled, input_minor_log, input_cooking_scaled])[0, 0]
    
    # Normalize the score to be between min_score and max_score
    normalized_score = score * (max_score - min_score) + min_score
    
    # Clip the score to ensure it's within the original range
    clipped_score = np.clip(normalized_score, min_score, max_score)
    
    return -clipped_score  # Minimize cost function to maximize the score

# Constraint for the sum of proportions to be 1
def constraint_sum(params):
    return np.sum(params[:len(major_ingredient_columns)]) - 1

# Initial guess for optimization (adjust as needed)
initial_guess = np.array([0.1] * len(major_ingredient_columns) + [0.01] * len(minor_ingredient_columns) + [180, 10, 0])  # Example values for temperature, duration, and order

# Optimization
result = minimize(predict_score, initial_guess, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint_sum})

# Best recipe result
if result.success:
    best_recipe = result.x
    recipe_df = pd.DataFrame([best_recipe[:len(major_ingredient_columns)]], columns=major_ingredient_columns)
    
    # Base de calcul : 14.26% d'œuf correspond à 53g
    oeuf_percent = recipe_df["Oeuf"].values[0] * 100
    oeuf_weight = 53  # 53g correspond à 14.26%
    reference_percent = oeuf_percent
    
    # Calculer le ratio de conversion pour tous les ingrédients
    conversion_factor = oeuf_weight / reference_percent

    # Calculer les quantités en grammes
    best_recipe_grams = {}
    recipe_text = []
    total_weight = 0
    
    for ingredient in major_ingredient_columns:
        proportion_percent = recipe_df[ingredient].values[0] * 100
        best_recipe_grams[ingredient] = proportion_percent * conversion_factor
        total_weight += best_recipe_grams[ingredient]
        ingredient_result = f"{ingredient}: {best_recipe_grams[ingredient]:.2f}g"
        recipe_text.append(ingredient_result)
        print(ingredient_result)

    # Ajouter les ingrédients mineurs
    for i, ingredient in enumerate(minor_ingredient_columns):
        amount = best_recipe[len(major_ingredient_columns) + i]
        amount_in_grams = amount * conversion_factor * 100  # Conversion en grammes
        best_recipe_grams[ingredient] = amount_in_grams
        total_weight += amount_in_grams
        ingredient_result = f"{ingredient}: {amount_in_grams:.2f}g"
        recipe_text.append(ingredient_result)
        print(ingredient_result)

    # Calculer les ratios pour le fichier CSV
    csv_recipe = {ingredient: round(grams / total_weight, 9) for ingredient, grams in best_recipe_grams.items()}

    # Ajouter les paramètres de cuisson
    for i, param in enumerate(cooking_parameter_columns):
        value = best_recipe[len(major_ingredient_columns) + len(minor_ingredient_columns) + i]
        if param == "Ordre":
            order_index = int(round(value))
            order = int_to_order[order_index]
            param_result = f"{param}: {order}"
            csv_recipe[param] = order
        else:
            param_result = f"{param}: {value:.2f}"
            csv_recipe[param] = round(value, 9)
        recipe_text.append(param_result)
        print(param_result)

    # Conversion de la note sur 5
    predicted_score = -predict_score(best_recipe)
    note_result = f"Note prédite: {predicted_score * 5:.2f} / 5.00"
    print(note_result)
    recipe_text.append(note_result)
    
    # Générer le nom de fichier avec la date et l'heure actuelles
    now = datetime.now()
    file_name = now.strftime("meilleure-recette_%Y%m%d-%H%M.txt")

    # Écrire les résultats dans un fichier texte
    with open(file_name, 'w') as f:
        f.write("\n".join(recipe_text))

    print(f"Les résultats ont été enregistrés dans le fichier : {file_name}")

    # Préparer les données pour l'export CSV
    csv_recipe["Note"] = "??"
    
    # Charger le fichier CSV existant
    existing_data = pd.read_csv('recettes_cookies.csv', sep=';')
    
    # Ajouter la nouvelle recette
    new_recipe = pd.DataFrame([csv_recipe])
    updated_data = pd.concat([existing_data, new_recipe], ignore_index=True)
    
    # Sauvegarder le fichier CSV mis à jour
    updated_data.to_csv('recettes_cookies.csv', sep=';', index=False, float_format='%.9f')
    
    print("La nouvelle recette a été ajoutée au fichier recettes_cookies.csv")

else:
    print("Optimization failed")