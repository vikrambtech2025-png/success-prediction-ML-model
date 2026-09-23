# Success Prediction ML Model

A **machine learning model that predicts the success of an ML/data project** from dataset characteristics — rows, columns, missing values, duplicates, and the numeric/categorical column mix. Trained on a synthetic dataset with a Random Forest classifier, served through a Streamlit app.

## How it works

1. `train_model.py` generates synthetic training data: dataset features (`n_rows`, `n_cols`, `n_missing`, `n_duplicated`, `num_cols`, `cat_cols`) → success label.
2. Trains a **RandomForestClassifier** and saves it with joblib as `model.pk`.
3. `app.py` loads the model and predicts a success score for new dataset profiles.

## Quick start

```bash
pip install -r requirements.txt
python train_model.py     # retrain (optional — model.pk ships with the repo)
streamlit run app.py
```

## Stack

Streamlit · scikit-learn · pandas · joblib

## License

No license specified — for learning/reference use.