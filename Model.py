import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm

def build_lstm_model(X_train_full):
    """
    Builds and returns a Bidirectional LSTM model for light curve forecasting.

    Parameters
    ----------
    X_train_full : np.ndarray
        The combined training feature array, used to define input shape.

    Returns
    -------
    model : keras.Sequential
        Compiled Bidirectional LSTM model.
    """
    input_shape = (1, X_train_full.shape[1])
    model = Sequential(name="Bidirectional_LSTM_Model")

    # === 1st Bidirectional LSTM Layer ===
    model.add(Bidirectional(
        LSTM(64, return_sequences=True,
             kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
             kernel_constraint=max_norm(2.0)),
        input_shape=input_shape))
    model.add(Dropout(0.1))

    # === 2nd Bidirectional LSTM Layer ===
    model.add(Bidirectional(
        LSTM(64, return_sequences=False)))
    model.add(Dropout(0.1))

    # === Dense Layers ===
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(90))  # 30 time steps × 3 filters or 30 × 6 = 180 for Rubin

    return model


# Example usage:
# X_train_full = np.concatenate([X_train, chirp_train], axis=1)
# X_train_reshaped = X_train_full.reshape((X_train_full.shape[0], 1, X_train_full.shape[1]))
# model = build_lstm_model(X_train_full)
# model.summary()