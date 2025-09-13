import pandas as pd
import trueskill
from collections import defaultdict

# Initialize TrueSkill environment (Bayesian rating system)
ts = trueskill.TrueSkill(draw_probability=0.00)  # 0% draws for FPS style games


class BayesianSkillRating:
    def __init__(self):
        self.ratings = defaultdict(ts.Rating)  # Player skill ratings

    def update_match(self, team_a, team_b, outcome):
        """
        team_a: list of player usernames in Team A
        team_b: list of player usernames in Team B
        outcome: "A" if Team A wins, "B" if Team B wins
        """
        ratings_a = [self.ratings[p] for p in team_a]
        ratings_b = [self.ratings[p] for p in team_b]

        if outcome == "A":
            new_ratings_a, new_ratings_b = ts.rate([ratings_a, ratings_b], ranks=[0, 1])
        else:
            new_ratings_a, new_ratings_b = ts.rate([ratings_a, ratings_b], ranks=[1, 0])

        for i, player in enumerate(team_a):
            self.ratings[player] = new_ratings_a[i]
        for i, player in enumerate(team_b):
            self.ratings[player] = new_ratings_b[i]

    def get_leaderboard(self):
        leaderboard = []
        for player, rating in self.ratings.items():
            skill_estimate = rating.mu - 3 * rating.sigma  # Conservative rating
            leaderboard.append(
                {
                    "player": player,
                    "mu": rating.mu,
                    "sigma": rating.sigma,
                    "skill_estimate": skill_estimate,
                }
            )
        return sorted(leaderboard, key=lambda x: x["skill_estimate"], reverse=True)


def process_dataset(file_path):
    df = pd.read_csv(file_path)

    rating_system = BayesianSkillRating()

    # Group matches
    for match_id, group in df.groupby("Match_ID"):
        team_a = group[group["Team"] == "Team_A"]["Username"].tolist()
        team_b = group[group["Team"] == "Team_B"]["Username"].tolist()

        if not team_a or not team_b:
            continue

        # Outcome decided by total kills
        team_a_kills = group[group["Team"] == "Team_A"]["Kills"].sum()
        team_b_kills = group[group["Team"] == "Team_B"]["Kills"].sum()
        outcome = "A" if team_a_kills > team_b_kills else "B"

        rating_system.update_match(team_a, team_b, outcome)

    return rating_system


if __name__ == "__main__":
    # Load and process dataset
    rating_system = process_dataset("Game_Win-Lost/DataSet.csv")

    # Print leaderboard
    leaderboard = rating_system.get_leaderboard()
    print("=== Bayesian TrueSkill Leaderboard ===")
    for rank, player in enumerate(leaderboard, start=1):
        print(
            f"#{rank:<3} | {player['player']:<15} | μ={player['mu']:.2f}, σ={player['sigma']:.2f}, Skill={player['skill_estimate']:.2f}"
        )
