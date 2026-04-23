# -*- coding: utf-8 -*-
"""
=============================================================================
 维度二：语义内容特征分析 — LDA 与 BERTopic 的对比检验及可视化重构
=============================================================================
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import jieba.posseg as pseg
import logging
from gensim import corpora, models
from gensim.models import CoherenceModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from hdbscan import HDBSCAN

# ============================================================================
# 全局配置 & 科研配色体系
# ============================================================================
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用维度一制定的学术莫兰迪色卡
COLOR_PALETTE = ['#8EAAA4', '#D69882', '#7F6C92', '#89949D', '#C7C0C3', '#E1DFE2', '#B6B0B9', '#5E4D71', '#587E76', '#C05C3F']
sns.set_palette(sns.color_palette(COLOR_PALETTE))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'dimension2')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'stopwords', 'chinese_stopwords.txt')

INPUT_FILE = os.path.join(DATA_DIR, 'search_comments_2026-04-15_cleaned.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'douyin_data_with_topics.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)

K_RANGE = range(4, 7)            
MIN_TEXT_LENGTH = 5

# ============================================================================
# 分词与预处理
# ============================================================================
jieba.setLogLevel(logging.WARNING)
ai_custom_words = [
    'deepseek', 'chatgpt', 'gpt', 'claude', 'gemini', 'grok',
    'token', 'agent', 'llm', 'transformer', 'mcp', 'aigc', 'agi',
    '大模型', '大语言模型', '提示词', '智能体', '生成式ai',
    '人工智能', '机器学习', '深度学习', '神经网络', '注意力机制',
    '上下文', '豆包', '文心一言', '通义千问', '即梦',
    '科大讯飞', '字节跳动', 'openai', 'anthropic',
    '开源模型', '闭源模型', '多模态', '人形机器人', '脑机接口'
]
for word in ai_custom_words:
    jieba.add_word(word)

with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f if line.strip())
stopwords.update(['', ' ', '\n', '\r', '\t', '的', '了', '是', '我', '你', '在', '也', '就', '不', '有', '和'])

ALLOWED_POS = {'n', 'nr', 'ns', 'nt', 'nz', 'nl', 'ng', 'v', 'vn', 'vd', 'a', 'ad', 'an', 'eng', 'l', 'i'}

def tokenize_and_filter(text):
    text = text.lower()
    words = pseg.cut(text)
    filtered = []
    for word, flag in words:
        w = word.strip()
        if w in ['ds', 'deep', 'deep seek']: w = 'deepseek'
        if w in stopwords or flag not in ALLOWED_POS: continue
        if len(w) < 2 and not w.isascii(): continue
        if re.match(r'^[\d\s\W]+$', w) and not re.match(r'^[a-z]+$', w): continue
        if len(w) >= 2 or (w.isascii() and len(w) >= 1 and w.isalpha()):
            filtered.append(w)
    return filtered

# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 70)
    print("  维度二：语义内容特征分析 — LDA 与 BERTopic 对比检验与静态重绘")
    print("=" * 70)

    # ---------------- 1. 数据读取 ----------------
    print("\n[1] 数据预处理...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['content'])
    df['content'] = df['content'].astype(str)
    df_clean = df[df['content'].str.len() > MIN_TEXT_LENGTH].copy()
    
    print("  → 正在分词...")
    df_clean['tokens'] = df_clean['content'].apply(tokenize_and_filter)
    df_clean = df_clean[df_clean['tokens'].apply(len) > 0].copy()
    texts = df_clean['tokens'].tolist()
    print(f"  ✓ 有效文档数：{len(df_clean)}")

    # ---------------- 2. LDA 建模 ----------------
    print("\n[2] LDA 建模与寻找最优 K...")
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    coherence_scores = []
    lda_models = {}
    
    # 限制 LDA 输出以防刷屏
    for k in K_RANGE:
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, random_state=42, passes=5)
        cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
        score = cm.get_coherence()
        coherence_scores.append(score)
        lda_models[k] = lda_model
        print(f"    K={k:2d} → c_v={score:.4f}")

    best_idx = np.argmax(coherence_scores)
    best_k = list(K_RANGE)[best_idx]
    best_score = coherence_scores[best_idx]
    best_lda = lda_models[best_k]

    # 图 1: LDA Coherence 折线图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_RANGE), coherence_scores, 'o-', color='#D69882', linewidth=2.5, markersize=8)
    ax.axvline(x=best_k, color='#89949D', linestyle='--', label=f'Best K = {best_k}')
    ax.set_title('LDA 主题数 K 的 C_v 一致性得分演化', fontsize=14, fontweight='bold')
    ax.set_xlabel('话题数 K')
    ax.set_ylabel('C_v Score')
    ax.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'lda_coherence_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 2: LDA Top Words 条形组图
    cols = 2
    rows = (best_k + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < best_k:
            words_probs = best_lda.show_topic(i, topn=10)
            words = [w[0] for w in words_probs]
            probs = [w[1] for w in words_probs]
            sns.barplot(x=probs, y=words, ax=axes[i], color=COLOR_PALETTE[i % len(COLOR_PALETTE)])
            axes[i].set_title(f'LDA Topic {i}', fontweight='bold')
            axes[i].set_xlabel('Probability')
        else:
            fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lda_top_words.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------- 3. BERTopic 建模 ----------------
    print("\n[3] BERTopic 深度语义建模...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    vectorizer_model = CountVectorizer(tokenizer=tokenize_and_filter, max_features=3000, min_df=3, max_df=0.5)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=60, min_samples=15, metric='euclidean', prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        nr_topics=10
    )
    
    docs = df_clean['content'].tolist()
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - (1 if -1 in topic_info['Topic'].values else 0)
    print(f"  ✓ 成功识别 {n_topics} 个深度主题")

    # [优化补充 1] 计算 BERTopic 的 C_v 一致性得分
    topic_words_list = []
    for t_id in topic_info['Topic']:
        if t_id == -1: continue
        top_words = [w for w, _ in topic_model.get_topic(t_id)]
        topic_words_list.append(top_words)
    
    ber_cm = CoherenceModel(topics=topic_words_list, texts=texts, dictionary=dictionary, coherence='c_v', processes=1)
    bertopic_cv_score = ber_cm.get_coherence()
    print(f"  ✓ BERTopic 自动一致性得分(C_v): {bertopic_cv_score:.4f}")

    # 图 3: BERTopic Top Words Bar Chart
    top_n = min(6, n_topics) # 画前6个话题
    fig, axes = plt.subplots((top_n + 1)//2, 2, figsize=(14, 4 * ((top_n + 1)//2)))
    axes = axes.flatten()
    topic_idx = 0
    for t_id in topic_info['Topic']:
        if t_id == -1 or topic_idx >= top_n: continue
        words_probs = topic_model.get_topic(t_id)
        words = [w[0] for w in words_probs][:10]
        probs = [w[1] for w in words_probs][:10]
        sns.barplot(x=probs, y=words, ax=axes[topic_idx], color=COLOR_PALETTE[topic_idx % len(COLOR_PALETTE)])
        # Label parsing
        short_label = "_".join(words[:3])
        axes[topic_idx].set_title(f'Cluster {t_id}: {short_label}', fontweight='bold')
        axes[topic_idx].set_xlabel('c-TF-IDF Score')
        topic_idx += 1
        
    for i in range(topic_idx, len(axes)): fig.delaxes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bertopic_barchart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 4: 选项A - UMAP 降维散点图
    print("  → 绘制 选项A: UMAP 降维散点图...")
    # 使用专门用于可视化的 2D UMAP
    umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    embeddings = embedding_model.encode(docs, show_progress_bar=False)
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    # Draw noise points first
    noise_idx = np.array(topics) == -1
    plt.scatter(embeddings_2d[noise_idx, 0], embeddings_2d[noise_idx, 1], color='#E1DFE2', s=5, alpha=0.3, label='Outliers (-1)')
    # Draw valid clusters
    for i, t_id in enumerate([t for t in topic_info['Topic'] if t != -1]):
        t_idx = np.array(topics) == t_id
        plt.scatter(embeddings_2d[t_idx, 0], embeddings_2d[t_idx, 1], s=15, alpha=0.8, color=COLOR_PALETTE[i % len(COLOR_PALETTE)], label=f'Topic {t_id}')
    plt.title('UMAP 2D 降维语义聚类云图 (BERTopic)', fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bertopic_umap_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 5: 选项B - 话题甜甜圈图
    print("  → 绘制 选项B: 话题结构甜甜圈图...")
    valid_topics = topic_info[topic_info['Topic'] != -1]
    labels = ["_".join([w[0] for w in topic_model.get_topic(t_id)[:3]]) for t_id in valid_topics['Topic']]
    sizes = valid_topics['Count']
    
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(sizes, autopct='%1.1f%%', startangle=90, pctdistance=0.85, 
                                       colors=COLOR_PALETTE[:len(sizes)], wedgeprops=dict(width=0.35, edgecolor='w'))
    plt.legend(wedges, labels, title="核心语义聚类簇", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('宏观信息生态：各核心话题占比分布 (Donut Chart)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bertopic_donut_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图 6: 选项C - 语义相似度热力图
    print("  → 绘制 选项C: 语义相似度热力图...")
    # Get topic embeddings (the 0th one is usually outlier, but let's carefully extract)
    valid_t_ids = [t for t in topic_info['Topic'] if t != -1]
    # We must construct a matrix of embeddings for valid topics
    t_embeds = []
    t_labels = []
    for t_id in valid_t_ids:
        # bertopic saves embeddings internally
        idx = topic_model._outliers + t_id
        if idx < len(topic_model.topic_embeddings_):
            t_embeds.append(topic_model.topic_embeddings_[idx])
            t_labels.append(f"T{t_id}: " + "_".join([w[0] for w in topic_model.get_topic(t_id)[:3]]))
    
    if len(t_embeds) > 1:
        sim_matrix = cosine_similarity(t_embeds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, xticklabels=t_labels, yticklabels=t_labels, cmap='mako_r', annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.title('BERTopic 簇间语义相似度矩阵 (Cosine Similarity)', fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'bertopic_similarity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ---------------- 4. 注意力扩散热度映射分析 ----------------
    print("\n[4] 生成 扩散热度 (注意力强度) 与话题关联分析...")
    # 组装数据
    topic_label_map = {}
    for t_id in set(topics):
        if t_id == -1:
            topic_label_map[t_id] = "Outliers"
        else:
            top_words = topic_model.get_topic(t_id)
            label = "_".join([w for w, _ in top_words[:3]])
            topic_label_map[t_id] = f"T{t_id}_{label}"

    df_clean['topic_id'] = topics
    df_clean['topic_label'] = df_clean['topic_id'].map(topic_label_map)
    df_export = df_clean.copy()
    
    # 过滤掉离群点进行热度统计
    df_valid = df_export[df_export['topic_id'] != -1]
    
    # 图 7: 话题 vs 点赞中位数 Barplot
    plt.figure(figsize=(10, 6))
    # We use median or mean with bootstrapping (sns does mean by default with 95% CI)
    # Using barplot shows the average attention intensity per topic
    sns.barplot(data=df_valid, y='topic_label', x='like_count', palette=COLOR_PALETTE, errorbar=('ci', 95), capsize=.1)
    plt.title('语义主题扩散热度分析 (注意力吸附效应)', fontweight='bold', fontsize=14)
    plt.xlabel('单条评论平均点赞数 (含 95% 置信区间误差棒)')
    plt.ylabel('深度学习子话题')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topic_attention_diffusion.png'), dpi=300, bbox_inches='tight')
    plt.close()

    df_export.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "─" * 50)
    print("📋 模型对比摘要总结：")
    print(f"  {'指标':<25} {'LDA':>15} {'BERTopic':>15}")
    print(f"  {'─'*55}")
    print(f"  {'最优主题数(Clusters)':<25} {best_k:>15} {n_topics:>15}")
    print(f"  {'一致性得分(C_v)':<27} {best_score:>15.4f} {bertopic_cv_score:>15.4f}")
    print(f"  {'离群点文档占比':<26} {'0%':>15} {((len(df_clean)-len(df_valid))/len(df_clean)*100):>14.1f}%")
    print(f"  {'方法论底层':<25} {'概率生成':>15} {'语义嵌入映射':>15}")
    print("─" * 50)
    
    print("\n✅ 所有静态高清组图与数据输出均已完成！落盘于 results/dimension2/")

if __name__ == '__main__':
    main()
