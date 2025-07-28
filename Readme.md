# Rivals - Player Skill Estimation System (Simple Elo Version)

This repository contains a prototype of a player skill estimation system for a competitive multiplayer game. This version uses the classic **Elo rating algorithm** for a simple and robust implementation.

## Architecture

The system's design is modular and straightforward:

- **`EloRating`**: A class dedicated to the core Elo calculations. It computes the expected outcome of a match and updates player ratings accordingly.
- **`Player`**: A simple data class to store a player's ID, username, and their current Elo rating.
- **`Match`**: A data class to hold information about a single match, including the players on each team and the final outcome.
- **`SkillRatingSystem`**: The main controller. It reads match data from a CSV, processes the matches sequentially, updates player ratings using the `EloRating` class, and provides simple methods to retrieve player data.

## Algorithm: Elo Rating System

I chose the Elo rating system for its **simplicity, widespread recognition, and effectiveness**.

### How It Works

1.  **Initial Rating**: Every new player starts with a base rating (e.g., 1500).
2.  **Expected Score**: Before a match, the system calculates the _expected score_ (or probability of winning) for each team. This is based on the difference between the teams' average ratings. A higher-rated team is expected to have a higher probability of winning.
3.  **Rating Update**: After the match, the ratings are updated.
    - If a player **wins**, their rating increases. Winning against a higher-rated opponent yields a larger rating gain.
    - If a player **loses**, their rating decreases. Losing to a lower-rated opponent results in a larger rating loss.
    - The **K-factor** is a constant that determines the magnitude of rating changes. A higher K-factor makes ratings change more quickly.

### Why Elo for this Prototype?

- **Interpretability**: The logic is easy to understand and explain. A player's rating is a single, intuitive number.
- **Robustness**: Elo is a battle-tested system used in many competitive games and sports, including chess.
- **Simplicity**: The implementation is much simpler than more complex Bayesian systems like Glicko-2, making it a great starting point for a prototype.

While this implementation uses team win/loss as the primary input, the Elo framework can be extended. For instance, individual performance could influence the K-factor on a per-match basis, allowing exceptional players to climb the ranks faster.

## How to Run

1.  **Prerequisites**:

    - Python 3.x
    - pandas library (`pip install pandas`)

2.  **Setup**:

    - Place the `DataSet.csv` file in the same directory as the Python script.

3.  **Execution**:
    - Run the Python script from your terminal:
      ```bash
      python your_script_name.py
      ```
    - The script will process the data and print:
      - The current rating of an example player.
      - A top-10 leaderboard.
