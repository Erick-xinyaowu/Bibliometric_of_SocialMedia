import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from wordcloud import WordCloud
import jieba
import re
import os
import json

# ==========================================
# 0. Environment Setup & Aesthetic Theme
# ==========================================
# Seaborn styling
sns.set_theme(style="whitegrid", context="talk")

# Must set font AFTER sns.set_theme as seaborn overrides rcParams
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Academic Color Palette Definitions (Darkened versions of user's references)
color_scatter_bg = '#798894'       # Darkened Slate
color_scatter_fg = '#56426C'       # Darkened Muted Purple
color_scatter_log = '#456E65'      # Darkened Morandi Teal
color_trendline = '#C25D42'        # Punchier Muted Coral Red
color_bar = '#456E65'              # Darkened Teal
color_lorenz = '#56426C'           # Darkened Purple

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

metrics = {}

# Utility function for Lorenz Curve
def plot_lorenz_curve(values, title, save_path):
    sorted_vals = np.sort(values)
    cum_vals = np.cumsum(sorted_vals)
    cum_vals = cum_vals / cum_vals[-1]
    cum_entities = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    
    cum_entities = np.insert(cum_entities, 0, 0)
    cum_vals = np.insert(cum_vals, 0, 0)
    
    B = np.trapz(cum_vals, cum_entities)
    gini = 1 - 2*B
    
    plt.figure(figsize=(9, 7))
    plt.plot(cum_entities, cum_vals, label=f'Lorenz Curve (Gini = {gini:.4f})', color=color_lorenz, linewidth=3)
    plt.plot([0, 1], [0, 1], linestyle='--', color='#7f8c8d', linewidth=2, label='Perfect Equality (Gini = 0)')
    plt.fill_between(cum_entities, cum_vals, cum_entities, alpha=0.15, color=color_lorenz)
    plt.title(f'洛伦兹曲线：{title}', pad=15, fontweight='bold')
    plt.xlabel('累积实体比例 (Cumulative Proportion of Entities)', labelpad=10)
    plt.ylabel('累积资源比例 (Cumulative Proportion of Values)', labelpad=10)
    plt.legend(frameon=True, shadow=True, facecolor='white')
    # Let seaborn handle grid, just render and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return gini

# ==========================================
# 1. Task 1: Zipf's Law & Word Visuals
# ==========================================
print("[1/2] Processing Text Semantic (Zipf's Law)...")
all_words = []
for text in df['content']:
    text = re.sub(r'[^\w\s]', '', text)
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip() not in stopwords and len(w.strip()) > 1]
    all_words.extend(words)

word_counts = pd.Series(all_words).value_counts().reset_index()
word_counts.columns = ['word', 'frequency']
word_counts['rank'] = range(1, len(word_counts) + 1)

fit_data_z = word_counts[word_counts['frequency'] >= 3]
log_rank = np.log10(fit_data_z['rank'])
log_freq = np.log10(fit_data_z['frequency'])
slope_z, intercept_z, r_value_z, p_value_z, _ = linregress(log_rank, log_freq)
metrics['Zipf'] = {'alpha': round(abs(slope_z), 4), 'R_squared': round(r_value_z**2, 4), 'P_value': p_value_z}

# 1.1 Zipf Scatter
plt.figure(figsize=(9, 7))
plt.scatter(log_rank, log_freq, color=color_scatter_fg, alpha=0.9, s=45, edgecolor='white', linewidth=0.5, label='实测频次 (Freq >= 3)')
plt.plot(log_rank, intercept_z + slope_z * log_rank, color=color_trendline, linewidth=2.5,
         label=f'拟合线: $\\log(f) = {slope_z:.2f}\\log(r) + {intercept_z:.2f}$\n$R^2 = {r_value_z**2:.4f}$')
plt.title('Zipf定律：短视频平台评论文本词频分布', pad=15, fontweight='bold')
plt.xlabel('Log10 (Rank - 排名)', labelpad=10)
plt.ylabel('Log10 (Frequency - 词频)', labelpad=10)
plt.legend(frameon=True, shadow=True, facecolor='white')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "zipf_distribution.png"), dpi=300)
plt.close()

# 1.2 Top 30 Bar Chart
top_30_w = word_counts.head(30)
plt.figure(figsize=(12, 7))
sns.barplot(x=top_30_w['word'], y=top_30_w['frequency'], color=color_bar, edgecolor='#2980b9')
plt.xticks(rotation=45, ha='right')
plt.title('Top 30 高频词汇断崖分布 (头部注意力聚焦)', pad=15, fontweight='bold')
plt.ylabel('核心词汇出现总频次', labelpad=10)
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "word_top30_bar.png"), dpi=300)
plt.close()

# 1.3 Word Cloud
word_dict = dict(zip(word_counts['word'], word_counts['frequency']))
# Swap to magma for beautiful academic contrast
wc = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf', width=1600, height=1000, 
               background_color='#1a1a1a', colormap='magma', max_words=200).generate_from_frequencies(word_dict)
plt.figure(figsize=(14, 9))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Semantic Focus: 评论空间核心注意力词云全景', fontsize=22, fontweight='bold', color='#333333', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "word_cloud.png"), dpi=300)
plt.close()

# 1.4 Lorenz
gini_z = plot_lorenz_curve(word_counts['frequency'].values, '词频注意力资源基尼分配特征', os.path.join(results_dir, "word_lorenz.png"))

# ==========================================
# 2. Task 2: Power Law (Like Count Distribution)
# ==========================================
print("[2/2] Processing Like Counts (Power-Law)...")
likes = df['like_count'].dropna()
likes_positive = likes[likes > 0]

like_freq = likes_positive.value_counts().reset_index()
like_freq.columns = ['like_count', 'frequency']
like_freq = like_freq.sort_values(by='like_count')

# Raw Fit
log_like = np.log10(like_freq['like_count'])
log_like_freq = np.log10(like_freq['frequency'])
slope_l, intercept_l, r_value_l, p_value_l, _ = linregress(log_like, log_like_freq)
metrics['PowerLaw_Likes_Raw'] = {'beta': round(abs(slope_l), 4), 'R_squared': round(r_value_l**2, 4)}

# 2.1 Power Law Raw Scatter
plt.figure(figsize=(9, 7))
plt.scatter(log_like, log_like_freq, color=color_scatter_bg, alpha=0.75, s=25, label='实测原始点赞分布 (离散噪声区)')
plt.plot(log_like, intercept_l + slope_l * log_like, color=color_trendline, linewidth=2.5,
         label=f'受噪线性拟合: $R^2 = {r_value_l**2:.4f}$')
plt.title('基础幂律分布拟合 (含离散长尾噪声)', pad=15, fontweight='bold')
plt.xlabel('Log10 (Like Count - 点赞数)', labelpad=10)
plt.ylabel('Log10 (Frequency - 评论频次)', labelpad=10)
plt.legend(frameon=True, shadow=True, facecolor='white')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "like_count_distribution.png"), dpi=300)
plt.close()

# Optimized Binning
min_like = likes_positive.min()
max_like = likes_positive.max()
bins = np.logspace(np.log10(min_like), np.log10(max_like), num=15)
counts, bin_edges = np.histogram(likes_positive, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths = bin_edges[1:] - bin_edges[:-1]
density = counts / bin_widths

valid = density > 0
bin_centers_v = bin_centers[valid]
density_v = density[valid]

log_x_bin = np.log10(bin_centers_v)
log_y_bin = np.log10(density_v)

slope_l_opt, intercept_l_opt, r_value_l_opt, p_value_l_opt, _ = linregress(log_x_bin, log_y_bin)
metrics['PowerLaw_Likes_Optimized'] = {'beta': round(abs(slope_l_opt), 4), 'R_squared': round(r_value_l_opt**2, 4)}

# 2.2 Optimized Scatter
plt.figure(figsize=(9, 7))
plt.scatter(log_like, log_like_freq, color=color_scatter_bg, alpha=0.4, s=20, label='原始离散噪声底色')
plt.scatter(log_x_bin, log_y_bin, color=color_scatter_log, alpha=1.0, s=80, edgecolor='white', linewidth=1, label='平滑还原密度点 (Log Bins Density)')
plt.plot(log_x_bin, intercept_l_opt + slope_l_opt * log_x_bin, color=color_trendline, linewidth=2.5,
         label=f'优化拟合线: $\\log(y) = {-abs(slope_l_opt):.2f}\\log(x) + {intercept_l_opt:.2f}$\n真实平滑 $R^2 = {r_value_l_opt**2:.4f}$')
plt.title('优化幂律分布：按对数几何分箱平滑校验', pad=15, fontweight='bold')
plt.xlabel('Log10 (Like Count - 点赞数)', labelpad=10)
plt.ylabel('Log10 (Frequency Density - 概率密度)', labelpad=10)
plt.legend(frameon=True, shadow=True, facecolor='white')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "like_count_dist_optimized.png"), dpi=300)
plt.close()

# 2.3 Lorenz
gini_l = plot_lorenz_curve(likes.values, '社交点赞资源分配极化现象', os.path.join(results_dir, "like_lorenz.png"))

# ==========================================
# 3. Export
# ==========================================
with open(os.path.join(results_dir, "dimension1_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Aesthetics update complete. Saved to results folder.")
