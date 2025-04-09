import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from joblib import dump, load
from tensorflow.keras import losses
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
from tensorflow.keras.optimizers import Adam 
from Model import build_lstm_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

# Load the data
lightCurve = pd.read_csv("/Users/np699/Library/CloudStorage/OneDrive-DrexelUniversity/Untitled Folder/Forecaster/Forecast KN LC/Data/lightcurveDataCom7988.csv")
features = pd.read_csv("/Users/np699/Library/CloudStorage/OneDrive-DrexelUniversity/Untitled Folder/Forecaster/Forecast KN LC/Data/featureDataCom7988_pAstro.csv")

# Drop rows from lightCurve where 'simulation_id' is in [1624, 43162, 97688]
ids_to_drop = [1624, 43162, 97688]
lightCurve = lightCurve[~lightCurve['simulation_id'].isin(ids_to_drop)]

# Drop unnecessary columns from the features dataframe
features = features.drop(["simulation_id", 'far', 'snr', 'longitude','latitude'], axis=1)

# Filter the light curve data
filtered_df = lightCurve[lightCurve['filter'].isin(['ztfg', 'ztfr', 'ztfi'])].copy()
filtered_df = filtered_df[['filter', 'mag']]

num_light_curves = features.shape[0]

# Ensure the data is in the correct order
filter_order = ['ztfg', 'ztfr', 'ztfi']
filtered_df['filter'] = pd.Categorical(filtered_df['filter'], categories=filter_order, ordered=True)
filtered_df = filtered_df.sort_index()

# Consistency check
total_points = len(filtered_df)
num_time_points = total_points // (num_light_curves * len(filter_order))

# Reshape the data
y = np.empty((num_light_curves, num_time_points * len(filter_order)), dtype=filtered_df['mag'].dtype)
ztfg_mags = filtered_df[filtered_df['filter'] == 'ztfg']['mag'].values.reshape(num_light_curves, num_time_points)
ztfr_mags = filtered_df[filtered_df['filter'] == 'ztfr']['mag'].values.reshape(num_light_curves, num_time_points)
ztfi_mags = filtered_df[filtered_df['filter'] == 'ztfi']['mag'].values.reshape(num_light_curves, num_time_points)

# Populate reshaped array
for i in range(num_time_points):
    y[:, i * 3 + 0] = ztfg_mags[:, i]
    y[:, i * 3 + 1] = ztfr_mags[:, i]
    y[:, i * 3 + 2] = ztfi_mags[:, i]

# Adjust num_time_points to 30 by removing the last 41 points
num_time_points = 30
y = y[:, :num_time_points * len(filter_order)]

# Time array (adjusted to 30 points)
t_min = 0.1
t_max = 6.0
dt = 0.2
time_single = np.linspace(t_min, t_max, num_time_points)

# Standardize the feature data
feature_scaler = RobustScaler()
X = feature_scaler.fit_transform(features)
dump(feature_scaler, 'feature_scaler.joblib')
# Standardize the target data
target_scaler = RobustScaler()
y = target_scaler.fit_transform(y)
dump(target_scaler, 'target_scaler.joblib')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape y data for LSTM input
y_train_reshaped = y_train.reshape((y_train.shape[0], num_time_points, len(filter_order)))
y_test_reshaped = y_test.reshape((y_test.shape[0], num_time_points, len(filter_order)))

# Reshape X data for LSTM input
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

input_shape = (1, X_train.shape[1])  # Replace with your actual input shape
model = build_lstm_model(input_shape)

# Compile the model with a lower learning rate and gradient clipping
optimizer = Adam(learning_rate=0.0003, clipnorm=1.0)  # Only clipnorm is used
model.compile(optimizer=optimizer, loss='mse')

# Implement early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# Create a TensorBoard callback
logdir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Train the model
history = model.fit(X_train_reshaped, y_train, 
                    epochs=300, batch_size=32, validation_split=0.3, 
                    verbose=1, callbacks=[early_stopping, lr_reduction])

# Save the trained model
dump(model, 'LSTM_model.joblib')
model.summary()

def predict_with_uncertainty(model, X, n_iter=1000):

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


# Print results
print(f"Test MSE with MC Dropout (inverted): {test_mse_mc:.4f}")
print(f"Test R² with MC Dropout (inverted): {test_r2_mc:.4f}")


# Plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()