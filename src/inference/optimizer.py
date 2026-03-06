import pulp
import pandas as pd

from src import config as _config

# Default platform config used when called outside the dashboard (e.g. bulk_backtest)
_DEFAULT_PLATFORM_CFG = _config.PLATFORM_CONFIGS['draftkings']


def generate_optimal_lineup(slate_df, platform_cfg=None):
    """
    Solves the salary-constrained lineup optimization problem.

    Parameters
    ----------
    slate_df : pd.DataFrame
        Players with 'Salary', 'Position', and 'Predicted_Pts' columns
        (already normalized to internal standard names by predict.py).
    platform_cfg : dict, optional
        Entry from config.PLATFORM_CONFIGS. Defaults to DraftKings if omitted.

    Returns
    -------
    dict: Contains 'status', 'lineup_df', 'total_salary', 'total_points'
    """
    if platform_cfg is None:
        platform_cfg = _DEFAULT_PLATFORM_CFG

    salary_cap   = platform_cfg['salary_cap']
    roster_slots = platform_cfg['roster_slots']
    total_slots  = platform_cfg['total_slots']

    if 'Predicted_Pts' not in slate_df.columns or 'Salary' not in slate_df.columns:
        return {"status": "Error: Missing required columns for optimization."}

    prob = pulp.LpProblem("WNBA_DFS_Optimizer", pulp.LpMaximize)

    player_vars = [pulp.LpVariable(f"player_{i}", cat="Binary") for i in range(len(slate_df))]

    # Objective: Maximize total predicted points
    prob += pulp.lpSum(player_vars[i] * slate_df.iloc[i]['Predicted_Pts'] for i in range(len(slate_df)))

    # Constraint 1: Exactly total_slots players
    prob += pulp.lpSum(player_vars) == total_slots

    # Constraint 2: Total salary <= cap
    prob += pulp.lpSum(player_vars[i] * slate_df.iloc[i]['Salary'] for i in range(len(slate_df))) <= salary_cap

    # Constraint 3: Position minimums (driven by platform roster_slots config)
    for position, min_count in roster_slots.items():
        if position == 'UTIL':
            continue  # UTIL fills naturally from remaining slots
        prob += pulp.lpSum(
            player_vars[i] for i in range(len(slate_df))
            if position in str(slate_df.iloc[i]['Position'])
        ) >= min_count

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

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
