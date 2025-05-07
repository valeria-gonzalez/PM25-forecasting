import numpy as np # Numerical operations
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM # Model
from tensorflow.keras.layers import Dense, Dropout # Layer
from tensorflow.keras.optimizers import Adam # Optimizer function
import optuna # Hyperparameter optimization

class MLPOptimizer:
  """
  Class for hyperparameter optimizing for PM2.5 Forecaster.
  """
  def __init__(self, input_dim:int, n_trials:int, output_nodes:int):
    self.input_dim = input_dim
    self.n_trials = n_trials
    self.study = None
    self.X_train = None
    self.y_train = None
    self.X_val = None
    self.y_val = None
    self.output_nodes = output_nodes

  def create_model(self, trial):
    model = Sequential()

    # Hyperparameter selection
    neurons = trial.suggest_categorical("neurons", [8, 10, 50])
    # neurons_2 = trial.suggest_categorical("neurons_2", [10, 50, 100])
    # dropout_rate = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])
    
    model.add(LSTM(neurons, 
               input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
               activation="relu"
               ))
    
    # model.add(Dropout(dropout_rate))
    
    # model.add(LSTM(neurons_2, activation="relu"))

    # Output layer
    model.add(Dense(self.output_nodes))

    # Compile model
    learning_rate = 0.001
    model.compile(optimizer=Adam(learning_rate), loss="mse")

    return model

  def objective(self, trial):
    model = self.create_model(trial)
    epochs = trial.suggest_categorical("epochs", [100, 50, 30])
    batch_size = trial.suggest_categorical("batch_size", [32, 10, 72, 100])

    # Train the model
    model.fit(
        self.X_train,
        self.y_train,
        epochs=epochs,
        validation_data=(self.X_val, self.y_val),
        verbose=0,
        batch_size=batch_size,
        )

    # Evaluate on validation data
    loss = model.evaluate(self.X_val, self.y_val, verbose=0)

    return loss

  def optimize(self, X_train:np.array, y_train:np.array, X_val, y_val:np.array):
    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val

    self.study = optuna.create_study(direction="minimize")
    self.study.optimize(self.objective, n_trials = self.n_trials)

    print("Best hyperparameters:", self.study.best_params)
    return self.study