import tensorflow as tf
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import random

print(f"TensorFlow version: {tf.__version__}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

@tf.keras.utils.register_keras_serializable()
def log_transform(x):
    return tf.math.log1p(x)

# Load the ingredient columns from JSON file
with open('ingredient_columns.json', 'r') as f:
    ingredient_data = json.load(f)

major_ingredients = ingredient_data['major_ingredient_columns']
minor_ingredients = ingredient_data['minor_ingredient_columns']
cooking_parameters = ingredient_data['cooking_parameter_columns']

# Load the CSV data
data = pd.read_csv('recettes_cookies.csv', sep=';')

# Extract unique orders from the 'Ordre' column
unique_orders = data['Ordre'].unique().tolist()

# Create a dictionary to map orders to integers
order_to_int = {order: i for i, order in enumerate(unique_orders)}

# Save the orders to a JSON file
with open('orders.json', 'w') as f:
    json.dump({"orders": unique_orders}, f, indent=2)

# Convert the 'Ordre' column to integers
data['Ordre'] = data['Ordre'].map(order_to_int)

# Separate input features and target
X_ingredients = data[major_ingredients + minor_ingredients]
X_cooking = data[cooking_parameters]
y = data['Note']

# Split the data into training and validation sets
random_state = random.randint(0, 100)
X_ingredients_train, X_ingredients_val, X_cooking_train, X_cooking_val, y_train, y_val = train_test_split(
    X_ingredients, X_cooking, y, test_size=0.2, random_state=random_state)

# Create scalers
ingredient_scaler = MinMaxScaler()
cooking_scaler = MinMaxScaler()

# Fit and transform the training data
X_ingredients_train_scaled = ingredient_scaler.fit_transform(X_ingredients_train[major_ingredients])
X_cooking_train_scaled = cooking_scaler.fit_transform(X_cooking_train)

# Transform the validation data
X_ingredients_val_scaled = ingredient_scaler.transform(X_ingredients_val[major_ingredients])
X_cooking_val_scaled = cooking_scaler.transform(X_cooking_val)

# Log transform minor ingredients
X_minor_train = np.log1p(X_ingredients_train[minor_ingredients])
X_minor_val = np.log1p(X_ingredients_val[minor_ingredients])

# Create the model
input_major = Input(shape=(len(major_ingredients),))
input_minor = Input(shape=(len(minor_ingredients),))
input_cooking = Input(shape=(len(cooking_parameters),))

x_major = Dense(32, activation='relu')(input_major)
x_minor = Dense(32, activation='relu')(input_minor)
x_cooking = Dense(16, activation='relu')(input_cooking)

combined = Concatenate()([x_major, x_minor, x_cooking])
x = Dense(64, activation='relu')(combined)
x = Dense(32, activation='relu')(x)
output = Dense(1)(x)

model = tf.keras.Model(inputs=[input_major, input_minor, input_cooking], outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit(
    [X_ingredients_train_scaled, X_minor_train, X_cooking_train_scaled], 
    y_train,
    validation_data=([X_ingredients_val_scaled, X_minor_val, X_cooking_val_scaled], y_val),
    epochs=300,
    callbacks=[early_stopping],
    batch_size=4
)

# Save the model
model.save('cookie_model.keras')

# Save the scalers
joblib.dump(ingredient_scaler, 'ingredient_scaler.joblib')
joblib.dump(cooking_scaler, 'cooking_scaler.joblib')

# Define the bounds for ingredients, cooking parameters, and score
columns_bounds = {}
for ingredient in major_ingredients + minor_ingredients:
    min_val, max_val = X_ingredients[ingredient].min(), X_ingredients[ingredient].max()
    columns_bounds[ingredient] = [float(min_val), float(max_val)]

for param in cooking_parameters:
    min_val, max_val = X_cooking[param].min(), X_cooking[param].max()
    columns_bounds[param] = [float(min_val), float(max_val)]

# Add min and max score to columns_bounds
columns_bounds['score'] = [float(y.min()), float(y.max())]

# Save the bounds to a JSON file
with open('columns_bounds.json', 'w') as f:
    json.dump(columns_bounds, f, indent=2)

# Save the order_to_int mapping
with open('order_mapping.json', 'w') as f:
    json.dump(order_to_int, f, indent=2)

# Optionally, plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Training completed and model saved.")
