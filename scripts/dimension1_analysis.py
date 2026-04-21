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

metrics = {}

# Utility function for Lorenz Curve
def plot_lorenz_curve(values, title, save_path):
    sorted_vals = np.sort(values)
    # cumulative proportion of value
    cum_vals = np.cumsum(sorted_vals)
    cum_vals = cum_vals / cum_vals[-1]
    
    # cumulative proportion of entities
    cum_entities = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    
    # prepend 0s for plotting line from origin
    cum_entities = np.insert(cum_entities, 0, 0)
    cum_vals = np.insert(cum_vals, 0, 0)
    
    # calculate Gini coefficient (Area between perfect equality line and Lorenz curve)
    B = np.trapz(cum_vals, cum_entities)
    gini = 1 - 2*B
    
    plt.figure(figsize=(8, 6))
    plt.plot(cum_entities, cum_vals, label=f'Lorenz Curve (Gini = {gini:.4f})', color='orange', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Equality (Gini = 0)')
    plt.fill_between(cum_entities, cum_vals, cum_entities, alpha=0.2, color='orange')
    plt.title(f'洛伦兹曲线：{title}')
    plt.xlabel('累积实体比例 (Cumulative Proportion of Entities)')
    plt.ylabel('累积资源比例 (Cumulative Proportion of Values)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return gini

# ==========================================
# 1. Task 1: Zipf's Law & Word Additions
# ==========================================
print("Processing Words...")
all_words = []
for text in df['content']:
    text = re.sub(r'[^\w\s]', '', text)
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip() not in stopwords and len(w.strip()) > 1]
    all_words.extend(words)

word_counts = pd.Series(all_words).value_counts().reset_index()
word_counts.columns = ['word', 'frequency']
word_counts['rank'] = range(1, len(word_counts) + 1)

word_counts.to_csv(os.path.join(results_dir, "word_frequency.csv"), index=False, encoding='utf-8-sig')

# Fit
fit_data_z = word_counts[word_counts['frequency'] >= 3]
log_rank = np.log10(fit_data_z['rank'])
log_freq = np.log10(fit_data_z['frequency'])
slope_z, intercept_z, r_value_z, p_value_z, _ = linregress(log_rank, log_freq)
metrics['Zipf'] = {'alpha': round(abs(slope_z), 4), 'R_squared': round(r_value_z**2, 4), 'P_value': p_value_z}

# 1.1 Zipf Scatter
plt.figure(figsize=(8, 6))
plt.scatter(log_rank, log_freq, color='blue', alpha=0.5, label='实测频次 (Freq >= 3)')
plt.plot(log_rank, intercept_z + slope_z * log_rank, color='red', 
         label=f'拟合线: $\\log(f) = {slope_z:.2f}\\log(r) + {intercept_z:.2f}$\n$R^2 = {r_value_z**2:.4f}$')
plt.title('Zipf定律：评论文本词频分布检验')
plt.xlabel('排名 (Log10 Rank)')
plt.ylabel('词频 (Log10 Frequency)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(results_dir, "zipf_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# 1.2 Top 30 Bar Chart
top_30_w = word_counts.head(30)
plt.figure(figsize=(10, 8))
plt.bar(top_30_w['word'], top_30_w['frequency'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Top 30 高频词汇柱状图 (直观长尾衰减)')
plt.ylabel('出现频次')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(results_dir, "word_top30_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# 1.3 Word Cloud
word_dict = dict(zip(word_counts['word'], word_counts['frequency']))
# font path is typical for standard windows SimHei
wc = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf', width=1200, height=800, 
               background_color='white', colormap='ocean', max_words=200).generate_from_frequencies(word_dict)
plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('评论高频词云图：语义注意力焦点', fontsize=18)
plt.savefig(os.path.join(results_dir, "word_cloud.png"), dpi=300, bbox_inches='tight')
plt.close()

# 1.4 Lorenz Curve
gini_z = plot_lorenz_curve(word_counts['frequency'].values, '词频资源非均衡分配 (不平等度)', os.path.join(results_dir, "word_lorenz.png"))
metrics['Zipf']['Gini'] = round(gini_z, 4)

# ==========================================
# 2. Task 2: Power Law (Like Count Distribution)
# ==========================================
print("Processing Likes...")
likes = df['like_count'].dropna()
likes_positive = likes[likes > 0]

like_freq = likes_positive.value_counts().reset_index()
like_freq.columns = ['like_count', 'frequency']
like_freq = like_freq.sort_values(by='like_count')

# Fit
log_like = np.log10(like_freq['like_count'])
log_like_freq = np.log10(like_freq['frequency'])
slope_l, intercept_l, r_value_l, p_value_l, _ = linregress(log_like, log_like_freq)
metrics['PowerLaw_Likes'] = {'beta': round(abs(slope_l), 4), 'R_squared': round(r_value_l**2, 4), 'P_value': p_value_l}

# 2.1 Power Law Scatter
plt.figure(figsize=(8, 6))
plt.scatter(log_like, log_like_freq, color='green', alpha=0.5, label='实测点赞分布')
plt.plot(log_like, intercept_l + slope_l * log_like, color='red', 
         label=f'拟合线: $\\log(y) = {slope_l:.2f}\\log(x) + {intercept_l:.2f}$\n$R^2 = {r_value_l**2:.4f}$')
plt.title('幂律分布：评论点赞数分布检验')
plt.xlabel('点赞数 (Log10 Like Count)')
plt.ylabel('评论出现的频数 (Log10 Frequency)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(results_dir, "like_count_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2.2 Lorenz Curve
# use raw likes (including 0s if they existed, but better 0+ included to show real inequality)
gini_l = plot_lorenz_curve(likes.values, '评论获赞非均衡分配 (巨星效应)', os.path.join(results_dir, "like_lorenz.png"))
metrics['PowerLaw_Likes']['Gini'] = round(gini_l, 4)

# ==========================================
# 3. Export
# ==========================================
with open(os.path.join(results_dir, "dimension1_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Visualizations generated successfully.")
