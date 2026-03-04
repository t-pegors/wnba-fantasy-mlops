"""
End-to-end pipeline smoke test using a simulated DraftKings slate.
Verifies: model loading → inference → optimization → valid lineup output.

Usage:
    python scripts/test_pipeline.py
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from src import config
from src.models.predict import run_inference
from src.inference.optimizer import generate_optimal_lineup

SLATE_PATH = config.SLATES_DATA_DIR / "simulated_2025_test.csv"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition

def run_test():
    print(f"\n{'='*60}")
    print("  WNBA MLOps Pipeline — End-to-End Smoke Test")
    print(f"{'='*60}\n")

    all_passed = True

    # ── 1. PRE-CONDITIONS ─────────────────────────────────────────
    print("[ 1 / 4 ] Pre-condition checks")
    ok = check("Simulated slate exists", SLATE_PATH.exists(), str(SLATE_PATH))
    all_passed &= ok
    ok = check("Production model exists", config.MODEL_PATH.exists(), str(config.MODEL_PATH))
    all_passed &= ok
    ok = check("Golden table exists", (config.PROCESSED_DATA_DIR / "training_features.csv").exists())
    all_passed &= ok

    if not all_passed:
        print("\n  Aborting — missing required files.\n")
        return

    # ── 2. SLATE LOADING ──────────────────────────────────────────
    print("\n[ 2 / 4 ] Slate loading & validation")
    dk_df = pd.read_csv(SLATE_PATH)
    check("Required DK columns present",
          all(c in dk_df.columns for c in ['Name', 'Salary', 'Position', 'TeamAbbrev', 'Game Info']),
          f"columns: {list(dk_df.columns)}")
    check("Player count", len(dk_df) >= 8, f"{len(dk_df)} players")
    guard_count = dk_df['Position'].apply(lambda p: 'G' in str(p)).sum()
    fwd_count   = dk_df['Position'].apply(lambda p: 'F' in str(p)).sum()
    check("Enough guards for lineup (≥2)", guard_count >= 2, f"{guard_count} guards")
    check("Enough forwards for lineup (≥3)", fwd_count >= 3, f"{fwd_count} forwards")
    print(f"\n  Slate summary:")
    for _, row in dk_df.iterrows():
        print(f"    {row['Name']:<25} {row['Position']:<5} ${row['Salary']:,}  {row['TeamAbbrev']}")

    # ── 3. INFERENCE ──────────────────────────────────────────────
    print("\n[ 3 / 4 ] XGBoost Inference")
    result_df = run_inference(dk_df.copy())

    check("Predicted_Pts column added", 'Predicted_Pts' in result_df.columns)
    check("No null predictions", result_df['Predicted_Pts'].notna().all())

    # Flag if all predictions are identical (suggests dummy fallback was used)
    unique_preds = result_df['Predicted_Pts'].nunique()
    using_real_model = unique_preds > 2
    check("Real model used (predictions vary)", using_real_model,
          f"{unique_preds} unique values — {'real model' if using_real_model else 'FALLBACK/dummy math detected'}")

    # Cold-start detection: players getting exactly 12.0 baseline
    fallback_players = result_df[result_df['Predicted_Pts'].round(1) == 12.0]['Name'].tolist()
    if fallback_players:
        print(f"  {WARN} Players at flat 12.0 baseline (no historical data found):")
        for name in fallback_players:
            print(f"       - {name}")
    else:
        print(f"  {PASS} All players matched to historical anchors")

    print(f"\n  Predictions:")
    pred_display = result_df[['Name', 'Position', 'Salary', 'Predicted_Pts']].sort_values(
        'Predicted_Pts', ascending=False)
    for _, row in pred_display.iterrows():
        print(f"    {row['Name']:<25} {row['Position']:<5} ${row['Salary']:,}  → {row['Predicted_Pts']:.1f} pts")

    # ── 4. OPTIMIZATION ───────────────────────────────────────────
    print("\n[ 4 / 4 ] Lineup Optimization (PuLP)")
    opt_result = generate_optimal_lineup(result_df)

    check("Optimizer status is Optimal", opt_result['status'] == 'Optimal', opt_result['status'])

    if opt_result['status'] == 'Optimal':
        lineup = opt_result['lineup_df']
        total_sal = opt_result['total_salary']
        total_pts = opt_result['total_points']

        check("Exactly 6 players selected", len(lineup) == 6, f"{len(lineup)} players")
        check("Under salary cap", total_sal <= config.SALARY_CAP,
              f"${total_sal:,} / ${config.SALARY_CAP:,}")

        g_in_lineup = lineup['Position'].apply(lambda p: 'G' in str(p)).sum()
        f_in_lineup = lineup['Position'].apply(lambda p: 'F' in str(p)).sum()
        check("≥2 Guards in lineup", g_in_lineup >= 2, f"{g_in_lineup} guards")
        check("≥3 Forwards in lineup", f_in_lineup >= 3, f"{f_in_lineup} forwards")

        print(f"\n  🏆 Optimal Lineup (${total_sal:,} | {total_pts:.1f} projected pts)")
        print(f"  {'Name':<25} {'Pos':<5} {'Salary':>8}  {'Proj Pts':>9}")
        print(f"  {'-'*52}")
        for _, row in lineup.sort_values('Predicted_Pts', ascending=False).iterrows():
            print(f"  {row['Name']:<25} {row['Position']:<5} ${row['Salary']:>7,}  {row['Predicted_Pts']:>8.1f}")

    # ── SUMMARY ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if opt_result['status'] == 'Optimal' and using_real_model:
        print("  RESULT: PASS — Full pipeline operational with real model.")
    elif opt_result['status'] == 'Optimal':
        print("  RESULT: PARTIAL — Optimizer works but model used fallback math.")
        print("          Check MODEL_PATH in config.py.")
    else:
        print("  RESULT: FAIL — Optimizer could not find a valid lineup.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_test()
