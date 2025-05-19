import math


# Elo rating update function
def update_elo(player1, player2, score1, score2, k=32) -> tuple:
    """
    Updates Elo ratings for two players based on the outcome of a match.

    This function calculates and updates the Elo ratings for two players
    after a match, using the standard Elo rating system formula.

    Parameters:
    player1 (dict): A dictionary containing the current Elo rating of the first player.
                    Must have a 'mu' key representing the player's rating.
    player2 (dict): A dictionary containing the current Elo rating of the second player.
                    Must have a 'mu' key representing the player's rating.
    score1 (float): The actual result of the match for player1.
                    1.0 represents a win, 0.5 a draw, and 0.0 a loss.
    score2 (float): The actual result of the match for player2.
                    1.0 represents a win, 0.5 a draw, and 0.0 a loss.
    k (int, optional): The K-factor, which determines the maximum change in rating.
                       Defaults to 32, which is standard for chess ratings.

    Returns:
    tuple: A tuple containing two dictionaries (player1, player2) with their updated Elo ratings.
           Each dictionary contains the updated 'mu' value representing the new Elo rating.
    """
    rating1 = player1['mu']
    rating2 = player2['mu']

    expected1 = 1 / (1 + math.pow(10, (rating2 - rating1) / 400))
    expected2 = 1 - expected1

    player1['mu'] = rating1 + k * (score1 - expected1)
    player2['mu'] = rating2 + k * (score2 - expected2)

    return player1, player2

def main():
    player1 = {'mu': 1500, 'rd': 350}  # Default Glicko values
    player2 = {'mu': 1500, 'rd': 350}

    # Update ratings
    player1, player2 = update_elo(player1, player2, 1, 0)

    print(f"Player 1: mu={player1['mu']:.2f}, rd={player1['rd']:.2f}")
    print(f"Player 2: mu={player2['mu']:.2f}, rd={player2['rd']:.2f}")

if __name__ == "__main__":
    main()
