import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def analyze_offline_model(model_path):
    print(f"📊 Loading local model from: {model_path}")
    
    # 1. Load the binary file directly into XGBoost
    model = xgb.Booster()
    model.load_model(model_path)
    
    # 2. Extract importance scores
    # 'weight' = number of times a feature is used to split the data
    importance = model.get_score(importance_type='weight')
    
    # 3. Convert to DataFrame and calculate percentage impact
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance_Score': list(importance.values())
    })
    
    # Sort and calculate relative importance (%)
    total_weight = importance_df['Importance_Score'].sum()
    importance_df['Relative_Importance_Pct'] = (importance_df['Importance_Score'] / total_weight) * 100
    importance_df = importance_df.sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

    # 4. Print Table to Terminal
    print("\n--- 📋 COMPLETE FEATURE IMPORTANCE TABLE ---")
    print(importance_df.to_string(index=False, formatters={'Relative_Importance_Pct': '{:,.2f}%'.format}))
    print("--------------------------------------------\n")

    # 5. Save Table as CSV
    report_dir = os.path.join(project_root, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, 'feature_importance_audit.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"💾 Full table saved to: {csv_path}")

    # 6. Plotting (Top 15 for readability)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance_Score', y='Feature', data=importance_df.head(15), palette='magma')
    plt.title('Top 15 WNBA Features (by Split Weight)')
    plt.xlabel('Number of Splits (Weight)')
    plt.tight_layout()
    
    plot_path = os.path.join(report_dir, 'feature_importance_plot.png')
    plt.savefig(plot_path)
    print(f"🎨 Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    # Ensure this points to the latest 'model.ubj' you downloaded from Run 7f510540
    local_file = os.path.join(os.path.dirname(__file__), 'model.ubj')
    
    if os.path.exists(local_file):
        analyze_offline_model(local_file)
    else:
        print(f"❌ Error: {local_file} not found. Please download model.ubj from DagsHub first.")