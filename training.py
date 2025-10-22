import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# 1. Load the data
# ============================================================
lightCurve = pd.read_csv("/Users/np699/Library/CloudStorage/OneDrive-DrexelUniversity/Untitled Folder/Forecaster/lightcurveDataCom7988.csv")
features = pd.read_csv("/Users/np699/Library/CloudStorage/OneDrive-DrexelUniversity/Untitled Folder/Forecaster/featureDataCom7988_pAstro_with_distmean_std.csv")
print("LightCurve shape:", lightCurve.shape)
print("Features shape before cleaning:", features.shape)
# Drop unnecessary columns from the features dataframe
features = features.drop(["simulation_id", 'far', 'snr', 'longitude','latitude', 'distance', 'chirp_mass'], axis=1)

# ============================================================
# 2. Drop bad simulation IDs
# ============================================================
ids_to_drop = [1624, 43162, 97688]
lightCurve = lightCurve[~lightCurve['simulation_id'].isin(ids_to_drop)]

# ============================================================
# 3. Separate chirp_mass_bin_edges and keep it as list/array
# ============================================================
if 'chirp_mass_bin_edges' in features.columns:
    features['chirp_mass_bin_edges'] = features['chirp_mass_bin_edges'].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
    )

# ============================================================
# 4. Drop unnecessary columns for scaling (keep sim_id & chirp_mass_bin_edges)
# ============================================================
drop_cols = ['simulation_id', 'chirp_mass_bin_edges']
columns_to_scale = [c for c in features.columns if c not in drop_cols]

# Split features into scalar part and list part
scalar_features = features[columns_to_scale].copy()
chirp_edges = np.stack(features['chirp_mass_bin_edges'].values)
print(f"Chirp-mass bin edges shape: {chirp_edges.shape} (each row = one array)")
print(scalar_features)
# ============================================================
# 5. Convert all scalar columns to numeric and fill NaN
# ============================================================
scalar_features = scalar_features.apply(pd.to_numeric, errors='coerce')
if scalar_features.isnull().any().any():
    print("⚠️ NaN values found — filling with column means.")
    scalar_features = scalar_features.fillna(scalar_features.mean())

# ============================================================
# 6. Scale only the scalar features
# ============================================================
feature_scaler = RobustScaler()
X_scaled = feature_scaler.fit_transform(scalar_features)
dump(feature_scaler, 'feature_scaler_O4.joblib')

# Combine scaled features + chirp_mass_bin_edges into one structure
# Option 1: keep as tuple (X_scaled, chirp_edges)
# Option 2: concatenate if you plan to treat them jointly later
X_combined = {
    'scalar': X_scaled,
    'chirp_mass_bin_edges': chirp_edges
}

print("✅ Scaled scalar features and preserved chirp_mass_bin_edges array.")
print("X_scaled shape:", X_scaled.shape)
print("chirp_edges shape:", chirp_edges.shape)

# ============================================================
# 7. Prepare light curves (same as before)
# ============================================================
filtered_df = lightCurve[lightCurve['filter'].isin(['ztfg', 'ztfr', 'ztfi'])].copy()
filtered_df = filtered_df[['filter', 'mag']]

num_light_curves = scalar_features.shape[0]
filter_order = ['ztfg', 'ztfr', 'ztfi']
filtered_df['filter'] = pd.Categorical(filtered_df['filter'], categories=filter_order, ordered=True)
filtered_df = filtered_df.sort_index()

total_points = len(filtered_df)
num_time_points = total_points // (num_light_curves * len(filter_order))
print("Number of light curves:", num_light_curves)
print("Number of time points:", num_time_points)

y = np.empty((num_light_curves, num_time_points * len(filter_order)), dtype=filtered_df['mag'].dtype)
ztfg_mags = filtered_df[filtered_df['filter'] == 'ztfg']['mag'].values.reshape(num_light_curves, num_time_points)
ztfr_mags = filtered_df[filtered_df['filter'] == 'ztfr']['mag'].values.reshape(num_light_curves, num_time_points)
ztfi_mags = filtered_df[filtered_df['filter'] == 'ztfi']['mag'].values.reshape(num_light_curves, num_time_points)

for i in range(num_time_points):
    y[:, i * 3 + 0] = ztfg_mags[:, i]
    y[:, i * 3 + 1] = ztfr_mags[:, i]
    y[:, i * 3 + 2] = ztfi_mags[:, i]

# ============================================================
# 8. Trim to 30 time points if needed
# ============================================================
num_time_points = 30
y = y[:, :num_time_points * len(filter_order)]

# ============================================================
# 9. Scale targets
# ============================================================
target_scaler = RobustScaler()
y_scaled = target_scaler.fit_transform(y)
dump(target_scaler, 'target_scaler_O4.joblib')

# ============================================================
# 10. Split and reshape for LSTM
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
chirp_train, chirp_test = train_test_split(chirp_edges, test_size=0.3, random_state=42)

y_train_reshaped = y_train.reshape((y_train.shape[0], num_time_points, len(filter_order)))
y_test_reshaped = y_test.reshape((y_test.shape[0], num_time_points, len(filter_order)))
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print("\nFinal Shapes:")
print("X_train_reshaped:", X_train_reshaped.shape)
print("y_train_reshaped:", y_train_reshaped.shape)
print("chirp_train:", chirp_train.shape)

# Concatenate before training
X_train_full = np.concatenate([X_train, chirp_train], axis=1)
X_test_full = np.concatenate([X_test, chirp_test], axis=1)

print("Combined feature shape:", X_train_full.shape)
X_train_reshaped = X_train_full.reshape((X_train_full.shape[0], 1, X_train_full.shape[1]))
X_test_reshaped = X_test_full.reshape((X_test_full.shape[0], 1, X_test_full.shape[1]))
input_shape=(1, X_train_full.shape[1])
model = build_lstm_model(input_shape)
model.summary()

# Compile the model with a lower learning rate and gradient clipping
optimizer = Adam(learning_rate=0.0003, clipnorm=2.0)  # Only clipnorm is used
model.compile(optimizer=optimizer, loss='mse')

# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# Create a TensorBoard callback
logdir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Train the model
history = model.fit(X_train_reshaped, y_train, 
                    epochs=300, batch_size=128, validation_split=0.3, 
                    verbose=1, callbacks=[early_stopping, lr_reduction])

def predict_with_uncertainty(model, X, n_iter=1000):
    """
    Perform MC Dropout predictions with TensorFlow 2.x.
    
    Args:
    - model: The Keras model with dropout layers.
    - X: Input data to make predictions on.
    - n_iter: Number of forward passes to perform.
    
    Returns:
    - mean_preds: Mean predictions.
    - uncertainty: Standard deviation of predictions (uncertainty).
    """
    
    # Perform n_iter forward passes and collect the predictions
    preds = [model(X, training=True) for _ in range(n_iter)]  # Dropout active during inference
    preds = np.array(preds)
    
    # Calculate mean and standard deviation of the predictions
    mean_preds = preds.mean(axis=0)
    uncertainty = preds.std(axis=0)
    
    return mean_preds, uncertainty

# Perform MC Dropout to predict with uncertainty
n_mc_samples = 1000 # Number of forward passes
mean_preds, uncertainty = predict_with_uncertainty(model, X_test_reshaped, n_iter=n_mc_samples)

# Reshape the mean predictions to match the shape used during scaling (num_samples, num_time_points * num_filters)
mean_preds_flat = mean_preds.reshape(mean_preds.shape[0], num_time_points * 3)  
# Invert the standardization for the mean predictions
mean_preds_inverted = target_scaler.inverse_transform(mean_preds_flat).reshape(mean_preds.shape[0], num_time_points, 3)

# Reshape uncertainty to match mean_preds_inverted shape
uncertainty_reshaped = uncertainty.reshape(uncertainty.shape[0], num_time_points, 3)
# Evaluate the model using the mean predictions

test_mse_mc = mean_squared_error(y_test.flatten(), mean_preds_flat.flatten())
test_r2_mc = r2_score(y_test.flatten(), mean_preds_flat.flatten())

print(f'Test MSE with MC Dropout: {test_mse_mc:.4f}, Test R² with MC Dropout: {test_r2_mc:.4f}')

# Reshape y_test to match the scaler's expected input (num_samples, num_time_points * 3)
y_test_flat = y_test.reshape(y_test.shape[0], num_time_points * 3)

# Inverse-transform y_test using the same scaler
y_test_inverted = target_scaler.inverse_transform(y_test_flat).reshape(y_test.shape[0], num_time_points, 3)

# Flatten both for metric computation
y_test_inverted_flat = y_test_inverted.flatten()
mean_preds_inverted_flat = mean_preds_inverted.flatten()

# Compute metrics on the inverse-transformed data
test_mse_mc = mean_squared_error(y_test_inverted_flat, mean_preds_inverted_flat)
test_r2_mc = r2_score(y_test_inverted_flat, mean_preds_inverted_flat)
mae = mean_absolute_error(y_test_inverted_flat, mean_preds_inverted_flat)

# Print results
print(f"Test MSE with MC Dropout (inverted): {test_mse_mc:.4f}")
print(f"Test R² with MC Dropout (inverted): {test_r2_mc:.4f}")
print(f"MAE with MC Dropout (inverted): {mae:.4f}")

# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
