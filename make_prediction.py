import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
import gc

# 1. Load Data
print("Loading data...")
train = pd.read_csv('playground-series-s6e5/train.csv')
test = pd.read_csv('playground-series-s6e5/test.csv')
sub = pd.read_csv('playground-series-s6e5/sample_submission.csv')

# 2. Advanced Feature Engineering
print("Feature Engineering...")

df_full = pd.concat([train.assign(is_train=1), test.assign(is_train=0, PitNextLap=-1)], ignore_index=True)

# 2.1 Basic interactions and ratios
df_full['TyreLife_x_LapTime'] = df_full['TyreLife'] * df_full['LapTime (s)']
df_full['TyreLife_x_Degradation'] = df_full['TyreLife'] * df_full['Cumulative_Degradation']
df_full['LapNumber_x_RaceProgress'] = df_full['LapNumber'] * df_full['RaceProgress']
df_full['LapTime_ratio_TyreLife'] = df_full['LapTime (s)'] / (df_full['TyreLife'] + 1)
df_full['Degradation_ratio_TyreLife'] = df_full['Cumulative_Degradation'] / (df_full['TyreLife'] + 1)

# 2.2 Grouped Statistics
driver_stats = df_full.groupby('Driver')[['LapTime (s)', 'Cumulative_Degradation', 'TyreLife']].mean().reset_index()
driver_stats.columns = ['Driver', 'Driver_mean_LapTime', 'Driver_mean_Degradation', 'Driver_mean_TyreLife']
df_full = df_full.merge(driver_stats, on='Driver', how='left')

race_stats = df_full.groupby('Race')[['LapTime (s)', 'Cumulative_Degradation', 'TyreLife']].mean().reset_index()
race_stats.columns = ['Race', 'Race_mean_LapTime', 'Race_mean_Degradation', 'Race_mean_TyreLife']
df_full = df_full.merge(race_stats, on='Race', how='left')

# 2.3 Categorical Encoding
df_full = pd.get_dummies(df_full, columns=['Compound'], drop_first=True)

for col in ['Race', 'Driver']:
    freq = train[col].value_counts() / len(train)
    df_full[f'{col}_freq'] = df_full[col].map(freq).fillna(0)

df_full.drop(['Race', 'Driver'], axis=1, inplace=True)

# 2.4 Split back to train and test
train_fe = df_full[df_full['is_train'] == 1].drop(['is_train'], axis=1).copy()
test_fe = df_full[df_full['is_train'] == 0].drop(['is_train', 'PitNextLap'], axis=1).copy()

features = [c for c in train_fe.columns if c not in ['id', 'PitNextLap']]

# 3. Model Training (Ensemble)
print("Training models...")
X = train_fe[features]
y = train_fe['PitNextLap'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3.1 XGBoost
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, use_label_encoder=False,
    eval_metric='logloss', early_stopping_rounds=50
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
xgb_preds_val = xgb_model.predict_proba(X_val)[:, 1]

# 3.2 LightGBM
print("Training LightGBM...")
lgb_model = LGBMClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, objective='binary', n_jobs=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_preds_val = lgb_model.predict_proba(X_val)[:, 1]

# 3.3 CatBoost
print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.03, depth=6, random_seed=42,
    eval_metric='Logloss', early_stopping_rounds=50, verbose=100
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_preds_val = cat_model.predict_proba(X_val)[:, 1]

# 4. Evaluation (Ensemble)
ensemble_preds_val = (xgb_preds_val + lgb_preds_val + cat_preds_val) / 3
print("\n--- Validation Results ---")
print(f"XGBoost Log Loss: {log_loss(y_val, xgb_preds_val):.5f} | AUC: {roc_auc_score(y_val, xgb_preds_val):.5f}")
print(f"LightGBM Log Loss: {log_loss(y_val, lgb_preds_val):.5f} | AUC: {roc_auc_score(y_val, lgb_preds_val):.5f}")
print(f"CatBoost Log Loss: {log_loss(y_val, cat_preds_val):.5f} | AUC: {roc_auc_score(y_val, cat_preds_val):.5f}")
print(f"Ensemble Log Loss: {log_loss(y_val, ensemble_preds_val):.5f} | AUC: {roc_auc_score(y_val, ensemble_preds_val):.5f}")

# 5. Prediction
print("\nMaking predictions on test set...")
X_test = test_fe[features]
xgb_preds_test = xgb_model.predict_proba(X_test)[:, 1]
lgb_preds_test = lgb_model.predict_proba(X_test)[:, 1]
cat_preds_test = cat_model.predict_proba(X_test)[:, 1]

ensemble_preds_test = (xgb_preds_test + lgb_preds_test + cat_preds_test) / 3

# 6. Submission
sub['PitNextLap'] = ensemble_preds_test
sub.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
