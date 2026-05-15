import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score
import gc

# 1. Load Data
print("Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')

# 2. Feature Engineering
print("Feature Engineering...")
def feature_engineering(df, is_train=True):
    df = df.copy()

    # Interaction terms
    df['TyreLife_x_LapTime'] = df['TyreLife'] * df['LapTime (s)']
    df['TyreLife_x_Degradation'] = df['TyreLife'] * df['Cumulative_Degradation']
    df['LapNumber_x_RaceProgress'] = df['LapNumber'] * df['RaceProgress']

    # Ratio terms
    df['LapTime_ratio_TyreLife'] = df['LapTime (s)'] / (df['TyreLife'] + 1)

    # Encoding
    # Frequency encoding based on the train distribution
    for col in ['Race', 'Driver', 'Compound']:
        freq = train[col].value_counts() / len(train)
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Compound'], drop_first=True)

    # Clean up non-numeric columns
    df.drop(['Race', 'Driver'], axis=1, inplace=True)

    return df

train_fe = feature_engineering(train, is_train=True)
test_fe = feature_engineering(test, is_train=False)

# Ensure columns match after encoding
for col in train_fe.columns:
    if col not in test_fe.columns and col != 'PitNextLap':
        test_fe[col] = 0

features = [c for c in train_fe.columns if c not in ['id', 'PitNextLap']]

# 3. Model Training
print("Training model...")
X = train_fe[features]
y = train_fe['PitNextLap']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# 4. Evaluation
preds_val = model.predict_proba(X_val)[:, 1]
print(f"Validation Log Loss: {log_loss(y_val, preds_val):.5f}")
print(f"Validation ROC AUC: {roc_auc_score(y_val, preds_val):.5f}")

# 5. Prediction
print("Making predictions...")
X_test = test_fe[features]
preds_test = model.predict_proba(X_test)[:, 1]

# 6. Submission
sub['PitNextLap'] = preds_test
sub.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
