import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample match data with anti-cheat features
data = pd.DataFrame({
    'player_id': np.repeat([1, 2, 3, 4], 3),
    'match_id': np.tile([101, 102, 103], 4),
    'wins': np.random.randint(0, 10, 12),
    'kills': np.random.randint(0, 20, 12),
    'time_played': np.random.randint(5, 60, 12),
    'opponent_repeat_count': np.random.randint(0, 5, 12),  # suspicious if high
    'match_duration_sec': np.random.randint(60, 600, 12),
    'bid_amount': np.random.randint(10, 100, 12),
    'device_overlap_score': np.random.choice([0, 0.5, 1], 12),
    'dispute_count': np.random.randint(0, 3, 12),
    'account_age_days': np.random.randint(1, 500, 12),
    'rank_label': [3, 2, 1]*4  # Ground truth rank in match (high = better)
})

# Define features and label
feature_cols = [
    'wins', 'kills', 'time_played', 'opponent_repeat_count',
    'match_duration_sec', 'bid_amount', 'device_overlap_score',
    'dispute_count', 'account_age_days'
]
X = data[feature_cols]
y = data['rank_label']
group = data.groupby('match_id').size().to_list()  # Group size = players per match

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
group_train = [len(X_train)]  # Single group for small example

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the ranking model
model = XGBRanker(objective='rank:pairwise', n_estimators=100, learning_rate=0.1)
model.fit(X_train_scaled, y_train, group=group_train)

# Predict scores
pred_scores = model.predict(X_test_scaled)

# Output ranking predictions
result = X_test.copy()
result['predicted_rank_score'] = pred_scores
print(result.sort_values(by='predicted_rank_score', ascending=False))
