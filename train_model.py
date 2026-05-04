import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Dummy data: rows, cols, missing, duplicates, num_cols, cat_cols
np.random.seed(42)
n_samples = 1000
X = np.random.rand(n_samples, 6) * 1000
X[:, 0] = np.random.randint(10, 10000, n_samples)   # n_rows
X[:, 1] = np.random.randint(2, 20, n_samples)      # n_cols
X[:, 2] = np.random.randint(0, 500, n_samples)     # n_missing
X[:, 3] = np.random.randint(0, 200, n_samples)     # n_duplicated
X[:, 4] = np.random.randint(1, 10, n_samples)      # num_cols
X[:, 5] = np.random.randint(1, 10, n_samples)      # cat_cols

# Labels: more rows + fewer missing → higher success chance
y_proba = (
    0.7 * (X[:, 0] / X[:, 0].max())           # more rows → higher
    - 0.5 * (X[:, 2] / X[:, 2].max())         # more missing → lower
    + np.random.normal(0, 0.1, n_samples)
)
y = (y_proba > 0.5).astype(int)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pk")
print("✅ Model saved as model.pk")