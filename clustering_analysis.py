# ########################### P-Card Meal Clustering Analysis ###########################
# Author: Mike Ziemba  |  Last Updated: 7 May 2025  |  Project: Per Diem Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# Constants & File Paths
# ----------------------------
PER_DIEM = 100
TRANSACTIONS_CSV = "dummy_pcard_transactions.csv"
CARDHOLDER_LIMITS_CSV = "dummy_pcardholder_limits.csv"
EMPLOYEE_LIST_CSV = "dummy_employee_list.csv"

# ----------------------------
# Load and Prepare Data
# ----------------------------
def load_data():
    txns = pd.read_csv(TRANSACTIONS_CSV, low_memory=False)
    cards = pd.read_csv(CARDHOLDER_LIMITS_CSV)
    emps = pd.read_csv(EMPLOYEE_LIST_CSV)

    for df in [txns, cards, emps]:
        id_col = 'Employee - ID' if 'Employee - ID' in df.columns else 'employee_id'
        df[id_col] = df[id_col].astype(str).str.strip().str.replace('.0', '', regex=False)

    emps = emps.rename(columns={
        'Employee - ID': 'employee_id',
        'Last Name': 'last_name',
        'First Name': 'first_name',
        'Job Title': 'job_title',
        'Company': 'company',
        'Employee Status': 'employee_status'
    })

    txns['transaction_date'] = pd.to_datetime(txns['transaction_date'], errors='coerce')
    txns['transaction_amount'] = pd.to_numeric(txns['transaction_amount'], errors='coerce')

    return txns, cards, emps

# ----------------------------
# Create Daily and Monthly Summaries
# ----------------------------
def summarize_meals(txns, cards, emps):
    meals_df = txns[txns['supplier_merchant_category_group_description'] == 'Eating And Drinking Places'].copy()
    meals_df = meals_df.merge(cards[['employee_id']], on='employee_id', how='inner')
    meals_df = meals_df.merge(emps[['employee_id', 'company']], on='employee_id', how='left')  # ensure 'company' is added

    meals_df['year_month'] = meals_df['transaction_date'].dt.to_period('M')

    daily = meals_df.groupby(['employee_id', 'transaction_date']).agg(
        daily_total=('transaction_amount', 'sum')
    ).reset_index()

    daily['over_per_diem'] = (daily['daily_total'] > PER_DIEM).astype(int)
    daily['year_month'] = daily['transaction_date'].dt.to_period('M')

    company_lookup = meals_df[['employee_id', 'transaction_date', 'company_y']].drop_duplicates()
    company_lookup = company_lookup.rename(columns={'company_y': 'company'})
    daily = daily.merge(company_lookup, on=['employee_id', 'transaction_date'], how='left')

    return meals_df, daily

# ----------------------------
# Create Monthly Summary Per Cardholder
# ----------------------------
def get_monthly_cardholder_summary(daily):
    summary = daily.groupby(['employee_id', 'year_month']).agg(
        total_meal_days=('transaction_date', 'count'),
        days_over_limit=('over_per_diem', 'sum'),
        avg_daily_meal_total=('daily_total', 'mean'),
        max_daily_meal_total=('daily_total', 'max'),
        company=('company', 'first')
    ).reset_index()

    summary['pct_days_over_limit'] = summary['days_over_limit'] / summary['total_meal_days']
    summary['went_over_per_diem'] = (summary['days_over_limit'] > 0).astype(int)

    return summary

# ----------------------------
# Perform KMeans Clustering
# ----------------------------
def perform_clustering(df, k=3):
    cluster_df = df[['total_meal_days', 'avg_daily_meal_total', 'max_daily_meal_total']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled)

    clustered = df.loc[cluster_df.index].copy()
    clustered['cluster'] = labels

    return clustered

# ----------------------------
# Visualize and Export Cluster Insights
# ----------------------------
def save_cluster_charts(df):
    sns.set_theme()

    charts = [
        ('cluster_scatter_avg_vs_days.png', 'Cardholder Meal Behavior by Cluster',
         'total_meal_days', 'avg_daily_meal_total', 'Set1'),
        ('cluster_boxplot_avg.png', 'Avg Daily Meal Total by Cluster',
         'cluster', 'avg_daily_meal_total', 'Set2'),
        ('cluster_boxplot_max.png', 'Max Daily Meal Total by Cluster',
         'cluster', 'max_daily_meal_total', 'Set3')
    ]

    for filename, title, x, y, palette in charts:
        plt.figure(figsize=(10, 6))
        if 'scatter' in filename:
            sns.scatterplot(data=df, x=x, y=y, hue='cluster', palette=palette)
        else:
            sns.boxplot(data=df, x=x, y=y, hue='cluster', palette=palette, legend=False)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

# ----------------------------
# Export Clustered Data
# ----------------------------
def export_cluster_groups(clustered_df):
    for c in sorted(clustered_df['cluster'].unique()):
        out = clustered_df[clustered_df['cluster'] == c]
        out.to_excel(f"cluster_{c}_summary.xlsx", index=False)

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    txns, cards, emps = load_data()
    meals_df, daily_df = summarize_meals(txns, cards, emps)
    monthly_df = get_monthly_cardholder_summary(daily_df)
    clustered_df = perform_clustering(monthly_df, k=3)
    clustered_df = clustered_df.merge(emps[['employee_id', 'job_title']], on='employee_id', how='left')
    save_cluster_charts(clustered_df)
    export_cluster_groups(clustered_df)