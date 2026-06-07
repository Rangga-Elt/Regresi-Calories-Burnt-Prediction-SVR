import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import gamma
from sklearn.svm import SVR
from google.colab import files
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve

# Load data
uploaded = files.upload()

# Baca file CSV atau Excel
file_name = list(uploaded.keys())[0]
if file_name.endswith('.csv'):
    df = pd.read_csv(file_name)
elif file_name.endswith('.xlsx'):
    df = pd.read_excel(file_name)
else:
    raise ValueError("Format file tidak didukung. Harap unggah file CSV atau Excel.")

# Prapemrosesan
# Konversi Gender ke numerik 
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['Calories'])
y = df['Calories']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat Pipeline dengan StandardScaler dan SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# GridSearchCV untuk Pencarian Parameter
# Parameter grid untuk GridSearchCV
param_grid = {
    'svr__C': [5, 1],
    'svr__epsilon': [1, 0.05],
    'svr__kernel': ['rbf'],
    'svr__tol': [1e-3, 1e-4],
    'svr__max_iter': [100000],
    'svr__gamma': ['scale'],
    'svr__shrinking': [True]
}

# GridSearchCV untuk mencari kombinasi parameter terbaik
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=20,
    verbose=True,
    n_jobs=-1
)

# Latih model dengan GridSearchCV
grid_search.fit(X_train, y_train)

# Tampilkan parameter terbaik dari GridSearchCV
print("\nParameter Terbaik dari GridSearchCV:")
print(grid_search.best_params_)

# Model terbaik dari GridSearchCV
best_model = grid_search.best_estimator_

# Evaluasi Model Akhir
# Prediksi pada set pengujian
y_pred = best_model.predict(X_test)

# Evaluasi model menggunakan metrik
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nHasil Evaluasi Model Akhir:")
print(f"MAE: {mae:.7f}")
print(f"MSE: {mse:.7f}")
print(f"RMSE: {rmse:.7f}")
print(f"R² Score: {r2:.7f}")

# Cross-validation untuk evaluasi tambahan
# RMSE
cv_scores_rmse = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores_rmse)

# MAE
cv_scores_mae = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mae_scores = -cv_scores_mae  # Convert to positive values

# R-squared
cv_scores_r2 = cross_val_score(best_model, X, y, cv=5, scoring='r2')

# MSE
cv_scores_mse = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores_mse

print("\nHasil Cross-Validation:")

print(f"RMSE Rata-rata: {cv_rmse_scores.mean():.7f}")
print(f"Standar Deviasi RMSE: {cv_rmse_scores.std():.7f}")

print(f"MAE Rata-rata: {cv_mae_scores.mean():.7f}")
print(f"Standar Deviasi MAE: {cv_mae_scores.std():.7f}")

print(f"R-squared Rata-rata: {cv_scores_r2.mean():.7f}")
print(f"Standar Deviasi R-squared: {cv_scores_r2.std():.7f}")

print(f"MSE Rata-rata: {cv_mse_scores.mean():.7f}")
print(f"Standar Deviasi MSE: {cv_mse_scores.std():.7f}")

# Visualisasi Kurva Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit (y=x)')
plt.title('Actual vs Predicted Calories')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.legend()
plt.grid(True)
plt.show()

# Visualisasi Learning Curve
def plot_learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    # Hitung rata-rata dan standar deviasi
    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='red')
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='green')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='red', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validation Score')
    plt.xlabel("Training Examples")
    plt.ylabel("Mean Squared Error)
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
