import pandas as pd
import math
from collections import defaultdict

class EloRating:
    def __init__(self, k_factor=32):
        self.k_factor = k_factor
    def get_expected_score(self, rating_a, rating_b):
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    def update_rating(self, rating, expected_score, actual_score):
        return rating + self.k_factor * (actual_score - expected_score)

class CheatDetector:
    def __init__(self, headshot_threshold=0.75, accuracy_threshold=0.80, kd_ratio_threshold=20.0):
        self.headshot_threshold = headshot_threshold
        self.accuracy_threshold = accuracy_threshold
        self.kd_ratio_threshold = kd_ratio_threshold
    def check_match_performance(self, player_stats):
        flags = []
        kills = player_stats.get('Kills', 0)
        deaths = player_stats.get('Deaths', 0)
        headshots = player_stats.get('Headshots', 0)
        accuracy = player_stats.get('Accuracy', 0)
        if kills > 10 and (headshots / kills) > self.headshot_threshold:
            flags.append(f"Suspiciously high headshot rate: {headshots/kills:.2%}")
        if accuracy > self.accuracy_threshold:
            flags.append(f"Suspiciously high accuracy: {accuracy:.2%}")
        if kills >= 20 and deaths <= 1:
            kd_ratio = kills / max(1, deaths)
            if kd_ratio > self.kd_ratio_threshold:
                flags.append(f"Extreme K/D ratio: {kd_ratio:.1f}")
        return flags

class Player:
    def __init__(self, username):
        self.username = username
        self.player_ids = set()
        self.rating = 1500
        self.match_history = []

class Team:
    def __init__(self, name):
        self.name = name
        self.players = set()
        self.rating = 1500

class Match:
    def __init__(self, match_id, team_a_players, team_b_players, outcome):
        self.match_id = match_id
        self.team_a_players = team_a_players
        self.team_b_players = team_b_players
        self.outcome = outcome

class SkillRatingSystem:
    def __init__(self, data_file):
        self.players = {}
        self.teams = {}
        self.matches = {}
        self.elo = EloRating()
        self.cheat_detector = CheatDetector()
        self.flagged_players = defaultdict(list)
        self._load_and_process_data(data_file)
        self._calculate_final_team_ratings()
    def get_player(self, username, player_id):
        if username not in self.players:
            self.players[username] = Player(username)
        self.players[username].player_ids.add(player_id)
        return self.players[username]
    def get_team(self, team_name):
        if team_name not in self.teams and "Solo" not in team_name:
            self.teams[team_name] = Team(team_name)
        return self.teams.get(team_name)
    def _load_and_process_data(self, data_file):
        try:
            df = pd.read_csv(data_file)
        except FileNotFoundError:
            print(f"Error: The file {data_file} was not found.")
            return
        for match_id, group in df.groupby('Match_ID'):
            team_a_players = []
            team_b_players = []
            match_team_a_name = group[group['Team'] == 'Team_A']['Team_Name'].iloc[0]
            match_team_b_name = group[group['Team'] == 'Team_B']['Team_Name'].iloc[0]
            team_a = self.get_team(match_team_a_name)
            team_b = self.get_team(match_team_b_name)
            for _, row in group.iterrows():
                player = self.get_player(row['Username'], row['Player_ID'])
                if row['Team'] == 'Team_A' and team_a:
                    team_a.players.add(player.username)
                elif row['Team'] == 'Team_B' and team_b:
                    team_b.players.add(player.username)
                player_stats = row.to_dict()
                flags = self.cheat_detector.check_match_performance(player_stats)
                if flags:
                    self.flagged_players[player.username].append({
                        "match_id": match_id,
                        "flags": flags,
                        "stats": f"K: {row['Kills']}, D: {row['Deaths']}, HS: {row['Headshots']}, Acc: {row['Accuracy']:.2f}"
                    })
                if row['Team'] == 'Team_A':
                    team_a_players.append(player)
                else:
                    team_b_players.append(player)
            if not team_a_players or not team_b_players: continue
            team_a_kills = group[group['Team'] == 'Team_A']['Kills'].sum()
            team_b_kills = group[group['Team'] == 'Team_B']['Kills'].sum()
            outcome = 'A' if team_a_kills > team_b_kills else 'B'
            avg_rating_a = sum(p.rating for p in team_a_players) / len(team_a_players)
            avg_rating_b = sum(p.rating for p in team_b_players) / len(team_b_players)
            expected_a = self.elo.get_expected_score(avg_rating_a, avg_rating_b)
            expected_b = self.elo.get_expected_score(avg_rating_b, avg_rating_a)
            actual_a = 1 if outcome == 'A' else 0
            actual_b = 1 if outcome == 'B' else 0
            for player in team_a_players:
                player.rating = self.elo.update_rating(player.rating, expected_a, actual_a)
            for player in team_b_players:
                player.rating = self.elo.update_rating(player.rating, expected_b, actual_b)
    def _calculate_final_team_ratings(self):
        for team in self.teams.values():
            member_ratings = [self.players[username].rating for username in team.players if username in self.players]
            if member_ratings:
                team.rating = sum(member_ratings) / len(member_ratings)
    def get_full_player_leaderboard(self):
        sorted_players = sorted(self.players.values(), key=lambda p: p.rating, reverse=True)
        return [{"username": p.username, "rating": p.rating} for p in sorted_players]
    def get_all_teams(self):
        sorted_teams = sorted(self.teams.values(), key=lambda t: t.name)
        return [{"team_name": t.name, "rating": t.rating} for t in sorted_teams]
    def get_flagged_players_report(self):
        return self.flagged_players

if __name__ == '__main__':
    skill_system = SkillRatingSystem('Game_Win-Lost/DataSet.csv')
    print("--- FULL PLAYER SKILL RATING LEADERBOARD ---")
    full_leaderboard = skill_system.get_full_player_leaderboard()
    if full_leaderboard:
        for rank, player_info in enumerate(full_leaderboard, 1):
            print(f"#{rank:<3} | {player_info['username']:<15} | Rating: {player_info['rating']:.2f}")
    print("\n" + "="*50 + "\n")
    print("--- ALL TEAMS (GROUPED BY TEAM NAME) ---")
    all_teams = skill_system.get_all_teams()
    if all_teams:
        for team_info in all_teams:
            print(f"- Team Name: {team_info['team_name']:<20} | Average Rating: {team_info['rating']:.2f}")
    print("\n" + "="*50 + "\n")
    print("--- CHEAT DETECTION REPORT ---")
    flagged_report = skill_system.get_flagged_players_report()
    if flagged_report:
        for username, flags in sorted(flagged_report.items()):
            print(f"Player: {username} was flagged in {len(flags)} match(es):")
            for flag_info in flags:
                print(f"  - Match ID: {flag_info['match_id']}")
                print(f"    Stats: {flag_info['stats']}")
                print(f"    Reasons: {', '.join(flag_info['flags'])}")
            print("-" * 20)