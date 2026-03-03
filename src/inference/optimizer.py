import pulp
import pandas as pd

def generate_optimal_lineup(slate_df, salary_cap=50000):
    """
    Takes a dataframe of players with 'Salary' and 'Predicted_Pts' and 
    solves the Knapsack Problem for DraftKings WNBA constraints.
    
    Returns:
        dict: Contains 'status', 'lineup_df', 'total_salary', 'total_points'
    """
    # Defensive check
    if 'Predicted_Pts' not in slate_df.columns or 'Salary' not in slate_df.columns:
        return {"status": "Error: Missing required columns for optimization."}

    # Initialize the PuLP Problem
    prob = pulp.LpProblem("WNBA_DFS_Optimizer", pulp.LpMaximize)
    
    # Create boolean variables for every player (1 if drafted, 0 if not)
    player_vars = [pulp.LpVariable(f"player_{i}", cat="Binary") for i in range(len(slate_df))]
    
    # Objective: Maximize total predicted points
    prob += pulp.lpSum(player_vars[i] * slate_df.iloc[i]['Predicted_Pts'] for i in range(len(slate_df)))
    
    # Constraint 1: Exactly 6 players drafted
    prob += pulp.lpSum(player_vars) == 6
    
    # Constraint 2: Total Salary <= Cap
    prob += pulp.lpSum(player_vars[i] * slate_df.iloc[i]['Salary'] for i in range(len(slate_df))) <= salary_cap
    
    # Constraint 3: Position minimums (DraftKings WNBA: 2G, 3F, 1 UTIL)
    prob += pulp.lpSum(player_vars[i] for i in range(len(slate_df)) if 'G' in str(slate_df.iloc[i]['Position'])) >= 2
    prob += pulp.lpSum(player_vars[i] for i in range(len(slate_df)) if 'F' in str(slate_df.iloc[i]['Position'])) >= 3
    
    # Solve silently
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Evaluate Results
    if pulp.LpStatus[prob.status] == 'Optimal':
        selected_indices = [i for i in range(len(slate_df)) if player_vars[i].varValue == 1.0]
        optimal_lineup = slate_df.iloc[selected_indices].copy()
        
        return {
            "status": "Optimal",
            "lineup_df": optimal_lineup,
            "total_salary": optimal_lineup['Salary'].sum(),
            "total_points": optimal_lineup['Predicted_Pts'].sum()
        }
    else:
        return {"status": "Infeasible: Could not find valid lineup under cap constraints."}