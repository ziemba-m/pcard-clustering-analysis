# P-Card Meal Clustering Analysis

This project segments corporate cardholders based on their meal spending behavior using KMeans clustering. It identifies distinct cardholder profiles that can support internal audit, policy design, or targeted follow-up.

## Objective
- Categorize cardholders by spending patterns (frequency, averages, max)
- Help understand behavioral clusters for more strategic compliance interventions

## Tools Used
- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- KMeans Clustering
- Feature Engineering and Standardization

## Cluster Features
- `total_meal_days`: # of meal transactions in a month
- `avg_daily_meals_total`: Average daily meal spending
- `max_daily_meal_total`: Highest single-day total

## Outputs
- `outputs/cluster_scatter_avg_vs_days.png`: Cluster distribution by frequency and average spend
- `outputs/cluster_boxplot_avg.png`: Cluster-wise distribution of avg spend
- `outputs/cluster_boxplot_max.png`: Cluster-wise max spend comparison
- Clustered summaries in `cluster_0_summary.xlsx`, etc.

## How to Run
1. Place your P-Card datasets in a `/data` folder
2. Run `clustering_analysis.py`
3. Review cluster visualizations and Excel exports in `/outputs`

## Notes
This analysis is meant for exploratory segmentation and is not predictive. It complements the risk modeling project by providing unsupervised insights.

---

*Author: Mike Ziemba*
