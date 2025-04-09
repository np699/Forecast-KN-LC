from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm

def build_lstm_model(input_shape):
    """
    Builds and returns the LSTM model.

    Parameters:
    - input_shape: tuple, the shape of the input data (timesteps, features).

    Returns:
    - model: the compiled LSTM model.
    """
    model = Sequential(name="LSTM_Model")

    # First Bidirectional LSTM Layer with Batch Normalization and Dropout
    model.add(Bidirectional(LSTM(400, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0),
                                 input_shape=input_shape),
                            name="Bidirectional_LSTM_1"))
    # model.add(BatchNormalization(name="BatchNormalization_1"))
    model.add(Dropout(0.1, name="Dropout"))  # Increased dropout rate

    # Second Bidirectional LSTM Layer with Batch Normalization
    model.add(Bidirectional(LSTM(300, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_2"))
    model.add(Dropout(0.1))

    # Third Bidirectional LSTM Layer with Batch Normalization
    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_3"))
    model.add(Dropout(0.1))

    # Fourth Bidirectional LSTM Layer with Batch Normalization
    model.add(Bidirectional(LSTM(150, activation='relu', return_sequences=False,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 kernel_constraint=max_norm(3.0)),
                            name="Bidirectional_LSTM_4"))
    # model.add(BatchNormalization(name="BatchNormalization_2"))
    model.add(Dropout(0.1))  # Increased dropout rate

    # Intermediate dense layers
    model.add(Dense(200, activation='relu', name="Dense_Intermediate"))
    model.add(Dropout(0.05))
    model.add(Dense(100, activation='relu', name="Dense_Intermediate1"))
    model.add(Dropout(0.05))

    # Dense output layer
    model.add(Dense(90, name="Dense_Output"))

    return model