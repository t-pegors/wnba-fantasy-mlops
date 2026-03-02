import pulp
import pandas as pd
from src import config

def solve_lineup(players_df):
    """
    Optimizes a WNBA lineup with flexible Utility slot logic.
    """
    prob = pulp.LpProblem("WNBA_Lineup_Optimization", pulp.LpMaximize)
    
    # Decision Variables
    player_vars = pulp.LpVariable.dicts("Players", players_df.index, cat='Binary')
    
    # Objective: Maximize Predicted Points
    prob += pulp.lpSum([players_df.loc[i, 'predicted_pts'] * player_vars[i] for i in players_df.index])
    
    # Constraint 1: Salary Cap
    prob += pulp.lpSum([players_df.loc[i, 'salary'] * player_vars[i] for i in players_df.index]) <= config.SALARY_CAP
    
    # Constraint 2: Total Roster Size (e.g., 6)
    prob += pulp.lpSum([player_vars[i] for i in players_df.index]) == config.TOTAL_SLOTS
    
    # Constraint 3: Position Minimums
    # We require at least the minimum, the 'Utility' will naturally be filled by the next best value
    guards = players_df[players_df['position'] == 'G'].index
    forwards = players_df[players_df['position'] == 'F'].index
    
    prob += pulp.lpSum([player_vars[i] for i in guards]) >= config.ROSTER_SLOTS['G']
    prob += pulp.lpSum([player_vars[i] for i in forwards]) >= config.ROSTER_SLOTS['F']
    
    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    chosen_indices = [i for i in players_df.index if player_vars[i].varValue == 1]
    return players_df.loc[chosen_indices]