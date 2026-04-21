import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from wordcloud import WordCloud
import jieba
import re
import os
import json

# ==========================================
# 0. Environment Setup & Data Loading
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

base_dir = r"e:\Bibliometric_Analysis\Bibliometric_of_SocialMedia"
data_path = os.path.join(base_dir, "data", "search_comments_2026-04-15_cleaned.csv")
stop_words_path = os.path.join(base_dir, "stopwords", "chinese_stopwords.txt")

results_dir = os.path.join(base_dir, "results", "dimension1")
os.makedirs(results_dir, exist_ok=True)

df = pd.read_csv(data_path)
df['content'] = df['content'].astype(str)

with open(stop_words_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())
stopwords.update([" ", "", "\n", "\t", "的", "了", "是", "我", "你", "在", "也", "就", "不", "有", "和", "都", "这", "也"])

try:
    with open(os.path.join(results_dir, "dimension1_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
except:
    metrics = {}

# ==========================================
# Task 2: Power Law Optimization (Like Count)
# Using Logarithmic Binning to fix R^2 drop in sparse tail
# ==========================================
print("Processing Optimized Power Law for Like Count (Logarithmic Binning)...")
likes = df['like_count'].dropna()
likes_positive = likes[likes > 0]

# Standard calculation for head ratio (Top 10%)
likes_sorted = likes.sort_values(ascending=False).values
top_10_percent_comments_idx = int(len(likes_sorted) * 0.1)
top_10_likes_sum = likes_sorted[:top_10_percent_comments_idx].sum()
total_likes = likes.sum()
like_head_ratio = top_10_likes_sum / total_likes if total_likes > 0 else 0

# Optimized Fitting: Logarithmic Binning
min_like = likes_positive.min()
max_like = likes_positive.max()
# Create 15 logarithmically spaced bins
bins = np.logspace(np.log10(min_like), np.log10(max_like), num=15)

# Calculate histogram
counts, bin_edges = np.histogram(likes_positive, bins=bins)

# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# VERY IMPORTANT: Normalize by bin width to get Density (Prob / Width)
bin_widths = bin_edges[1:] - bin_edges[:-1]
density = counts / bin_widths

# Filter out empty bins
valid = density > 0
bin_centers_v = bin_centers[valid]
density_v = density[valid]

log_x_bin = np.log10(bin_centers_v)
log_y_bin = np.log10(density_v)

slope_l_opt, intercept_l_opt, r_value_l_opt, p_value_l_opt, _ = linregress(log_x_bin, log_y_bin)
beta_l_opt = abs(slope_l_opt)

metrics['PowerLaw_Likes_Optimized'] = {
    'beta': round(beta_l_opt, 4),
    'R_squared': round(r_value_l_opt**2, 4),
    'P_value': p_value_l_opt,
    'Top_10_Percent_Head_Ratio': round(like_head_ratio, 4),
    'Optimization_Method': 'Logarithmic Binning'
}

# 2.1 Optimized Power Law Scatter
plt.figure(figsize=(8, 6))

# Plot the raw scatter in pale grey to show the noise it had
raw_freq = likes_positive.value_counts().reset_index()
raw_freq.columns = ['like_count', 'frequency']
plt.scatter(np.log10(raw_freq['like_count']), np.log10(raw_freq['frequency']), 
            color='lightgray', alpha=0.3, label='原始离散数据分布 (散乱的长尾)', s=15)

# Plot the Logged Bin centers
plt.scatter(log_x_bin, log_y_bin, color='green', alpha=0.9, s=50, label='对数分箱平滑后密度 (Log Bins)')
plt.plot(log_x_bin, intercept_l_opt + slope_l_opt * log_x_bin, color='red', linewidth=2,
         label=f'优化拟合线: $\\log(Density) = {-beta_l_opt:.2f}\\log(x) + {intercept_l_opt:.2f}$\n$R^2 = {r_value_l_opt**2:.4f}, p={p_value_l_opt:.1e}$')

plt.title('幂律分布优化：评论点赞数对数分箱 (Logarithmic Binning)')
plt.xlabel('点赞数 (Log10 Like Count)')
plt.ylabel('出现密度 / Frequency Density (Log10)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(results_dir, "like_count_dist_optimized.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. Export
# ==========================================
with open(os.path.join(results_dir, "dimension1_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Optimization complete. Check 'like_count_dist_optimized.png' and json metrics.")
