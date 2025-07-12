# =======================================================
# Artificial Neural Network for Customer Churn Prediction
# Dataset: churn_modelling.csv
# =======================================================

# ==== Import Required Libraries ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf

# ==== Load Dataset ====
data = pd.read_csv('./dataset/churn_modelling.csv')
X = data.iloc[:, 3:-1]  # Exclude RowNumber, CustomerId, Surname, and Exited
y = data.iloc[:, -1]    # Target: Exited

# ==== Train-Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==== Preprocessing: Encoding & Normalization ====
ct = ColumnTransformer(
    transformers=[
        ('scaler', MinMaxScaler(), [0, 3, 4, 5, 6, 9]),       # Numerical columns
        ('encoder', OneHotEncoder(drop='first'), [1, 2])      # Categorical columns: Geography, Gender
    ],
    remainder='passthrough'
)

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

# ==== Build Artificial Neural Network ====
tf.random.set_seed(42)
input_shape = X_train_transformed.shape[1:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(6, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ==== Compile Model ====
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# ==== Train Model ====
history = model.fit(
    X_train_transformed, y_train,
    epochs=100,
    verbose=0  # Set to 1 to view progress per epoch
)

# ==== Plot Accuracy & Loss ====
pd.DataFrame(history.history).plot()
plt.title('Training Accuracy & Loss')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# ==== Predictions ====
y_pred = (model.predict(X_test_transformed).flatten() > 0.5).astype(int)

# ==== Evaluation ====
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# ==== Display Confusion Matrix ====
sns.heatmap(cm, annot=True, cmap='viridis', fmt='g', cbar=False)
plt.title(f'Confusion Matrix\nAccuracy: {acc:.2%}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
