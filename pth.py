import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# === BASIC INFORMATION ===

print("=== BASIC INFORMATION ===\n")
df = pd.read_excel(r'C:\Users\91798\Downloads\pythonca2.xlsx')

print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe(include='all'))
print("\nFirst 5 Rows:")
print(df.head())

# === DATA CLEANING ===

print("\n=== DATA CLEANING ===\n")
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')
df['Base MSRP'] = pd.to_numeric(df['Base MSRP'], errors='coerce')
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

print("Missing Values:\n")
print(df.isnull().sum())

# === ANALYSIS & VISUALIZATIONS ===

print("\n=== ANALYSIS & VISUALIZATIONS ===\n")

# 1. Bar Plot - Count of Vehicles by Make
print("Plot 1: Top 10 Electric Vehicle Makes")
plt.figure(figsize=(12, 6))
make_counts = df['Make'].value_counts().head(10)
sns.barplot(x=make_counts.index, y=make_counts.values)
plt.title('Top 10 Electric Vehicle Makes')
plt.xlabel('Make')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Histogram - Electric Range
print("Plot 2: Distribution of Electric Range")
plt.figure(figsize=(10, 6))
sns.histplot(df['Electric Range'].dropna(), bins=30, kde=True)
plt.title('Distribution of Electric Vehicle Range')
plt.xlabel('Electric Range (miles)')
plt.ylabel('Frequency')
plt.show()

# 3. Pair Plot
print("Plot 3: Pair Plot of Numeric Variables")
numeric_cols = ['Electric Range', 'Base MSRP', 'Model Year']
sns.pairplot(df[numeric_cols].dropna())
plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
plt.show()

# 4. Box Plot by Vehicle Type
print("Plot 4: Electric Range by Vehicle Type")
plt.figure(figsize=(12, 6))
sns.boxplot(x='Electric Vehicle Type', y='Electric Range', data=df)
plt.title('Electric Range Distribution by Vehicle Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Heatmap
print("Plot 5: Correlation Heatmap")
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

# 6. Scatter Plot
print("Plot 6: Electric Range vs Model Year")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Model Year', y='Electric Range', data=df, hue='Electric Vehicle Type')
plt.title('Electric Range vs Model Year')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# === Z-TEST ANALYSIS ===

print("\n=== Z-TEST ANALYSIS ===\n")
bev_data = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']['Electric Range'].dropna()
phev_data = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']['Electric Range'].dropna()

bev_mean = bev_data.mean()
phev_mean = phev_data.mean()
bev_std = bev_data.std()
phev_std = phev_data.std()
bev_n = len(bev_data)
phev_n = len(phev_data)

z_score = (bev_mean - phev_mean) / np.sqrt((bev_std**2 / bev_n) + (phev_std**2 / phev_n))
p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test

print(f"BEV Mean Range: {bev_mean:.2f} miles")
print(f"PHEV Mean Range: {phev_mean:.2f} miles")
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")
print("Conclusion:", "Statistically significant difference in electric range between BEVs and PHEVs." if p_value < 0.05 else "No statistically significant difference in electric range between BEVs and PHEVs.")

# === ADDITIONAL INSIGHTS ===

# 8. Count of Vehicles by Year
print("\nPlot 7: EV Count by Model Year")
plt.figure(figsize=(12, 6))
year_counts = df['Model Year'].value_counts().sort_index()
sns.barplot(x=year_counts.index, y=year_counts.values)
plt.title('Electric Vehicle Count by Model Year')
plt.xlabel('Model Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Box Plot - Electric Range by Top Makes
print("Plot 8: Electric Range Distribution by Make (Top 10)")
plt.figure(figsize=(12, 6))
top_makes = df['Make'].value_counts().head(10).index
sns.boxplot(x='Make', y='Electric Range', data=df[df['Make'].isin(top_makes)])
plt.title('Electric Range Distribution by Make (Top 10)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. CAFV Eligibility Count
print("Plot 9: CAFV Eligibility Status Count")
plt.figure(figsize=(10, 6))
eligibility_counts = df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'].value_counts()
sns.barplot(x=eligibility_counts.index, y=eligibility_counts.values)
plt.title('CAFV Eligibility Status Count')
plt.xlabel('Eligibility Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === ANALYSIS COMPLETE ===

print("\n=== ANALYSIS COMPLETE ===")
