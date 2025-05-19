import math


def update_glicko(player1, player2, score1, score2, tau=0.5) -> tuple:
    """
    Updates Glicko ratings for two players based on a 1v1 match result.

    This function implements the Glicko rating system to update the ratings of two players
    after a match. It takes into account the current ratings, rating deviations, and the
    match outcome to calculate new ratings and rating deviations for both players.

    Parameters:
    player1 (dict): A dictionary containing the current Glicko rating ('mu') and
                    rating deviation ('rd') of the first player.
    player2 (dict): A dictionary containing the current Glicko rating ('mu') and
                    rating deviation ('rd') of the second player.
    score1 (float): The score of player1 in the match. Typically 1 for a win,
                    0.5 for a draw, and 0 for a loss.
    score2 (float): The score of player2 in the match. Typically 1 for a win,
                    0.5 for a draw, and 0 for a loss.
    tau (float, optional): The system constant, representing the change in rating deviation
                           over time. Defaults to 0.5.

    Returns:
    tuple: A tuple containing two dictionaries, (player1, player2), with their updated
           Glicko ratings ('mu') and rating deviations ('rd').

    Note:
    The function modifies the input dictionaries in-place and also returns them.
    """
    q = math.log(10) / 400
    g_rd = lambda rd: 1 / math.sqrt(1 + (3 * q**2 * rd**2) / (math.pi**2))
    e = lambda mu, mu_j, g_rd_j: 1 / (1 + 10 ** (-g_rd_j * (mu - mu_j) / 400))

    g_rd1 = g_rd(player2['rd'])
    g_rd2 = g_rd(player1['rd'])
    e1 = e(player1['mu'], player2['mu'], g_rd1)
    e2 = e(player2['mu'], player1['mu'], g_rd2)

    d2_1 = 1 / (q**2 * g_rd1**2 * e1 * (1 - e1))
    d2_2 = 1 / (q**2 * g_rd2**2 * e2 * (1 - e2))

    # Convert to win/loss
    score1 = score1/(score1 + score2)
    score2 = 1 - score1

    # Rating updates
    new_mu1 = player1['mu'] + (q / ((1 / player1['rd']**2) + (1 / d2_1))) * g_rd1 * (score1 - e1)
    new_rd1 = math.sqrt(((1 / player1['rd']**2) + (1 / d2_1))**-1)

    new_mu2 = player2['mu'] + (q / ((1 / player2['rd']**2) + (1 / d2_2))) * g_rd2 * (score2 - e2)
    new_rd2 = math.sqrt(((1 / player2['rd']**2) + (1 / d2_2))**-1)

    # Apply rating deviation decay
    new_rd1 = min(math.sqrt(player1['rd']**2 + tau**2), new_rd1)
    new_rd2 = min(math.sqrt(player2['rd']**2 + tau**2), new_rd2)

    player1['mu'] = new_mu1
    player2['mu'] = new_mu2
    player1['rd'] = new_rd1
    player2['rd'] = new_rd2

    return player1, player2

def main():
    # Initialize Glicko ratings for two players
    player1 = {'mu': 1662, 'rd': 290}  # Default Glicko values
    player2 = {'mu': 1500, 'rd': 350}

    # Update ratings
    player1, player2 = update_glicko(player1, player2, 0.3, 0.7)

    print(f"Player 1: mu={player1['mu']:.2f}, rd={player1['rd']:.2f}")
    print(f"Player 2: mu={player2['mu']:.2f}, rd={player2['rd']:.2f}")

if __name__ == "__main__":
    main()
