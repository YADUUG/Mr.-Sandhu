
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Game_Win-Lost/DataSet.csv")

def engineer_features(df):
    df['KDA'] = (df['Kills'] + df['Assists']) / df['Deaths'].replace(0, 1)
    df['CombatScore'] = df['Kills']*2 + df['Assists']*1.5 + df['Headshots']*3
    df['PerMinScore'] = df['CombatScore'] / df['Time_Played_Min'].replace(0, 1)
    return df

data = engineer_features(data)

player_ratings = {pid: 1000 for pid in data['Player_ID'].unique()}

features = ['Kills', 'Deaths', 'Assists', 'Accuracy', 'Headshots', 'Objective_Score', 'Time_Played_Min', 'KDA', 'PerMinScore']
target = []
X = []

for _, row in data.iterrows():
    current_rating = player_ratings[row['Player_ID']]
    target.append(current_rating)
    X.append(row[features].values)

X = np.array(X)
y = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

def update_player_ratings(df, model):
    for _, row in df.iterrows():
        pid = row['Player_ID']
        features_vec = row[features].values.reshape(1, -1)
        new_rating = model.predict(features_vec)[0]
        old_rating = player_ratings.get(pid, 1000)
        player_ratings[pid] = 0.9 * old_rating + 0.1 * new_rating  # Smooth update

update_player_ratings(data, model)

def get_player_rating(pid):
    return player_ratings.get(pid, 1000)

def get_recent_matches(pid, n=5):
    return data[data['Player_ID'] == pid].sort_values(by='Match_ID', ascending=False).head(n)

def get_leaderboard(n=10):
    df = pd.DataFrame(list(player_ratings.items()), columns=['Player_ID', 'SkillRating'])
    return df.sort_values(by='SkillRating', ascending=False).head(n)

print("Leaderboard:\n", get_leaderboard())
