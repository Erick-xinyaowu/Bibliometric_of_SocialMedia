# -*- coding: utf-8 -*-


import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ============================================================================
# 全局配置
# ============================================================================
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'stopwords', 'chinese_stopwords.txt')

INPUT_FILE = os.path.join(DATA_DIR, 'search_comments_2026-04-15_cleaned.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'douyin_data_with_topics.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

K_RANGE = range(4, 13)  # K = 4 ~ 12
MIN_TEXT_LENGTH = 5

# ============================================================================
# 分词函数（定义在顶层，供 CountVectorizer 的 tokenizer 引用）
# ============================================================================
import jieba
import logging
jieba.setLogLevel(logging.WARNING)  # 抑制 jieba 的 "Building prefix dict" 噪音日志

# 添加 AI 领域自定义词汇 — 提高专业术语的分词准确性
ai_custom_words = [
    'deepseek', 'DeepSeek', 'chatgpt', 'ChatGPT', 'GPT', 'gpt',
    'claude', 'Claude', 'gemini', 'Gemini', 'grok', 'Grok',
    'token', 'Token', 'agent', 'Agent', 'LLM', 'llm',
    'transformer', 'Transformer', 'MCP', 'mcp',
    'AIGC', 'aigc', 'AGI', 'agi',
    '大模型', '大语言模型', '提示词', '智能体',
    '人工智能', '机器学习', '深度学习', '神经网络',
    '生成式AI', '生成式', '注意力机制',
    '向量', '词元', '上下文', '上下文窗口',
    '豆包', '文心一言', '通义千问', '即梦',
    '科大讯飞', '字节跳动', 'OpenAI', 'Anthropic',
    '开源模型', '闭源模型', '多模态',
    '人形机器人', '脑机接口', '自动驾驶',
    '斯皮尔伯格', '裘德洛', '奥斯卡',
    '数字人', '虚拟试穿', '语音合成',
    '提示词工程', 'prompt', 'Prompt',
    '意见领袖', '信息茧房', '算力',
]
for word in ai_custom_words:
    jieba.add_word(word)

# 加载停用词
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f if line.strip())
stopwords.update(['', ' ', '\n', '\r', '\t'])


def tokenize_and_filter(text):
    """
    中文分词 + 停用词过滤 + 长度过滤
    保留长度 >= 2 的有意义词汇（允许英文专业术语通过）
    """
    words = jieba.cut(text, cut_all=False)
    filtered = []
    for w in words:
        w = w.strip()
        if w in stopwords:
            continue
        if len(w) < 2 and not w.isascii():
            continue
        if re.match(r'^[\d\s\W]+$', w) and not re.match(r'^[a-zA-Z]+$', w):
            continue
        if len(w) >= 2 or (w.isascii() and len(w) >= 2):
            filtered.append(w)
    return filtered


# ============================================================================
# 主流程（Windows 兼容：包裹在 __main__ 保护块中）
# ============================================================================
def main():
    print("=" * 70)
    print("  维度二：语义内容特征分析 — LDA 与 BERTopic 对比检验")
    print("=" * 70)

    # ========================================================================
    # 第一部分：数据预处理
    # ========================================================================
    print("\n" + "─" * 50)
    print("📦 第一部分：数据预处理")
    print("─" * 50)

    print("\n[1.1] 读取评论数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  ✓ 原始数据：{len(df)} 条评论，涉及 {df['aweme_id'].nunique()} 条视频")

    print(f"\n[1.2] 数据清洗：过滤空值及长度 ≤ {MIN_TEXT_LENGTH} 字符的评论...")
    df = df.dropna(subset=['content'])
    df['content'] = df['content'].astype(str)
    df['text_length'] = df['content'].str.len()

    short_count = (df['text_length'] <= MIN_TEXT_LENGTH).sum()
    df_clean = df[df['text_length'] > MIN_TEXT_LENGTH].copy()
    print(f"  ✓ 过滤超短评论 {short_count} 条")
    print(f"  ✓ 保留有效评论 {len(df_clean)} 条 ({len(df_clean)/len(df)*100:.1f}%)")

    print("\n[1.3] jieba 中文分词（含自定义 AI 领域词典）...")
    print("  → 正在分词（可能需要 1~2 分钟）...")
    df_clean['tokens'] = df_clean['content'].apply(tokenize_and_filter)
    df_clean['tokens_str'] = df_clean['tokens'].apply(lambda x: ' '.join(x))

    empty_tokens = (df_clean['tokens'].apply(len) == 0).sum()
    df_clean = df_clean[df_clean['tokens'].apply(len) > 0].copy()
    print(f"  ✓ 分词完成，过滤空分词结果 {empty_tokens} 条")
    print(f"  ✓ 最终有效文档数：{len(df_clean)} 条")

    print("\n  📋 分词示例（前 3 条）：")
    for i, row in df_clean.head(3).iterrows():
        print(f"    原文: {row['content'][:60]}...")
        print(f"    分词: {row['tokens'][:15]}")
        print()

    # ========================================================================
    # 第二部分：LDA 主题建模
    # 使用 LdaModel（单线程）以兼容 Windows multiprocessing 限制
    # ========================================================================
    print("\n" + "─" * 50)
    print("📊 第二部分：LDA 主题建模")
    print("─" * 50)

    from gensim import corpora, models
    from gensim.models import CoherenceModel

    print("\n[2.1] 构建 gensim 词典与 BoW 语料库...")
    texts = df_clean['tokens'].tolist()
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f"  ✓ 词典大小：{len(dictionary)} 个唯一词汇")
    print(f"  ✓ 语料库大小：{len(corpus)} 篇文档")

    print(f"\n[2.2] 搜索最优话题数 K（范围：{K_RANGE.start} ~ {K_RANGE.stop - 1}）...")
    print("  → 这一步计算量较大，请耐心等待...")

    coherence_scores = []
    lda_models = {}

    for k in K_RANGE:
        # 使用 LdaModel 替代 LdaMulticore，避免 Windows spawn 问题
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            chunksize=100,
            passes=10,
            per_word_topics=True,
            alpha='auto',
            eta='auto'
        )

        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1  # 强制单进程，避免 Windows 子进程反复加载 jieba
        )
        score = coherence_model.get_coherence()
        coherence_scores.append(score)
        lda_models[k] = lda_model
        print(f"    K = {k:2d} → Coherence Score (c_v) = {score:.4f}")

    # 2.3 绘制 Coherence Score 折线图
    print("\n[2.3] 绘制 Coherence Score 折线图...")

    fig, ax = plt.subplots(figsize=(10, 6))
    k_values = list(K_RANGE)
    ax.plot(k_values, coherence_scores, 'o-', color='#2563EB', linewidth=2, markersize=8)

    best_idx = np.argmax(coherence_scores)
    best_k = k_values[best_idx]
    best_score = coherence_scores[best_idx]
    ax.axvline(x=best_k, color='#DC2626', linestyle='--', alpha=0.7, label=f'最优 K = {best_k}')
    ax.scatter([best_k], [best_score], color='#DC2626', s=150, zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f'K={best_k}\nc_v={best_score:.4f}',
                xy=(best_k, best_score),
                xytext=(best_k + 0.5, best_score + 0.01),
                fontsize=11, fontweight='bold', color='#DC2626',
                arrowprops=dict(arrowstyle='->', color='#DC2626'))

    ax.set_xlabel('话题数 (K)', fontsize=13)
    ax.set_ylabel('一致性得分 (Coherence Score, c_v)', fontsize=13)
    ax.set_title('LDA 模型一致性得分随话题数 K 的变化', fontsize=15, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    coherence_fig_path = os.path.join(OUTPUT_DIR, 'lda_coherence_scores.png')
    fig.savefig(coherence_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存：{coherence_fig_path}")

    # 2.4 最优 K 输出
    print(f"\n[2.4] 使用最优 K = {best_k} 的 LDA 模型输出话题关键词：")
    best_lda = lda_models[best_k]

    print(f"\n  {'='*60}")
    print(f"  LDA 最优模型结果（K = {best_k}，c_v = {best_score:.4f}）")
    print(f"  {'='*60}")

    for topic_id in range(best_k):
        words = best_lda.show_topic(topic_id, topn=10)
        word_str = ' | '.join([f"{w}({p:.3f})" for w, p in words])
        print(f"\n  话题 {topic_id}: {word_str}")

    # 2.5 pyLDAvis 可视化
    print(f"\n[2.5] 生成 pyLDAvis 交互式可视化...")
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis

        vis_data = gensimvis.prepare(best_lda, corpus, dictionary, sort_topics=False)
        lda_vis_path = os.path.join(OUTPUT_DIR, 'lda_visualization.html')
        pyLDAvis.save_html(vis_data, lda_vis_path)
        print(f"  ✓ 已保存：{lda_vis_path}")
    except Exception as e:
        print(f"  ⚠ pyLDAvis 可视化生成失败：{e}")
        print("  → 将跳过此步骤，不影响后续分析")

    # ========================================================================
    # 第三部分：BERTopic 主题建模
    # ========================================================================
    print("\n" + "─" * 50)
    print("🤖 第三部分：BERTopic 主题建模")
    print("─" * 50)

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    print("\n[3.1] 加载多语言嵌入模型 paraphrase-multilingual-MiniLM-L12-v2...")
    print("  → 首次运行需下载模型（约 420MB），后续使用缓存")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("  ✓ 嵌入模型加载成功")

    print("\n[3.2] 配置 BERTopic 模型...")

    vectorizer_model = CountVectorizer(
        tokenizer=tokenize_and_filter,
        max_features=5000,
        min_df=3,
        max_df=0.5
    )

    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric='euclidean',
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True,
        nr_topics='auto'
    )

    print("\n[3.3] 拟合 BERTopic 模型（可能需要几分钟）...")
    docs = df_clean['content'].tolist()
    topics, probs = topic_model.fit_transform(docs)

    print("\n[3.4] BERTopic 主题分布结果：")
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - (1 if -1 in topic_info['Topic'].values else 0)
    print(f"\n  ✓ 共识别出 {n_topics} 个主题（不含离群点 Topic -1）")

    bertopic_outliers = topic_info[topic_info['Topic'] == -1]['Count'].values[0] if -1 in topic_info['Topic'].values else 0
    print(f"  ✓ 离群点文档数：{bertopic_outliers}")

    print(f"\n  {'='*70}")
    print(f"  BERTopic 主题分布")
    print(f"  {'='*70}")
    print(f"  {'Topic':>6} | {'Count':>6} | {'Name':<50}")
    print(f"  {'─'*6}-+-{'─'*6}-+-{'─'*50}")
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        name = str(row.get('Name', ''))[:50]
        print(f"  {topic_id:>6} | {count:>6} | {name}")

    print(f"\n  {'='*70}")
    print(f"  各主题 Top-10 关键词")
    print(f"  {'='*70}")
    for topic_id in topic_info['Topic'].values:
        if topic_id == -1:
            continue
        try:
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                words_str = ' | '.join([f"{w}({s:.3f})" for w, s in topic_words[:10]])
                print(f"\n  话题 {topic_id}: {words_str}")
        except Exception:
            pass

    # 3.5 BERTopic 可视化
    print(f"\n[3.5] 生成 BERTopic 可视化图表...")

    try:
        fig_topics = topic_model.visualize_topics()
        topics_vis_path = os.path.join(OUTPUT_DIR, 'bertopic_topics.html')
        fig_topics.write_html(topics_vis_path)
        print(f"  ✓ 主题距离图已保存：{topics_vis_path}")
    except Exception as e:
        print(f"  ⚠ 主题距离图生成失败：{e}")

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=min(10, n_topics), n_words=10)
        barchart_path = os.path.join(OUTPUT_DIR, 'bertopic_barchart.html')
        fig_barchart.write_html(barchart_path)
        print(f"  ✓ 关键词柱状图已保存：{barchart_path}")
    except Exception as e:
        print(f"  ⚠ 关键词柱状图生成失败：{e}")

    # ========================================================================
    # 第四部分：模型对比与结果导出
    # ========================================================================
    print("\n" + "─" * 50)
    print("📝 第四部分：模型对比与结果导出")
    print("─" * 50)

    print("\n[4.1] LDA 与 BERTopic 模型对比摘要：")
    print(f"\n  {'指标':<25} {'LDA':>15} {'BERTopic':>15}")
    print(f"  {'─'*55}")
    print(f"  {'识别主题数':<25} {best_k:>15} {n_topics:>15}")
    print(f"  {'一致性得分(c_v)':<24} {best_score:>15.4f} {'N/A':>15}")
    print(f"  {'离群点/未分类文档数':<22} {'0':>15} {bertopic_outliers:>15}")
    print(f"  {'方法论基础':<25} {'词袋+概率生成':>15} {'语义嵌入+密度聚类':>15}")

    # 4.2 关联 BERTopic 话题标签
    print("\n[4.2] 关联 BERTopic 话题标签到原始数据...")

    topic_label_map = {}
    for topic_id in set(topics):
        if topic_id == -1:
            topic_label_map[topic_id] = "离群点（未分类）"
        else:
            try:
                top_words = topic_model.get_topic(topic_id)
                if top_words:
                    label = "_".join([w for w, _ in top_words[:3]])
                    topic_label_map[topic_id] = f"Topic{topic_id}_{label}"
                else:
                    topic_label_map[topic_id] = f"Topic_{topic_id}"
            except Exception:
                topic_label_map[topic_id] = f"Topic_{topic_id}"

    df_clean['bertopic_topic_id'] = topics
    df_clean['topic_label'] = df_clean['bertopic_topic_id'].map(topic_label_map)

    # 4.3 导出
    print("\n[4.3] 导出带有话题标签的最终数据...")

    export_columns = [
        'comment_id', 'aweme_id', 'content', 'create_time', 'user_id',
        'like_count', 'sub_comment_count', 'ip_location', '所属地区',
        'bertopic_topic_id', 'topic_label'
    ]
    export_cols_available = [c for c in export_columns if c in df_clean.columns]
    df_export = df_clean[export_cols_available].copy()
    df_export.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"  ✓ 已保存：{OUTPUT_CSV}")
    print(f"  ✓ 数据行数：{len(df_export)}")
    print(f"  ✓ topic_label 缺失值数量：{df_export['topic_label'].isna().sum()}")

    # 4.4 话题分布统计
    print("\n[4.4] 话题分布统计：")
    topic_dist = df_export['topic_label'].value_counts()
    print(f"\n  {'话题标签':<45} {'文档数':>8} {'占比':>8}")
    print(f"  {'─'*63}")
    for label, count in topic_dist.items():
        pct = count / len(df_export) * 100
        print(f"  {label:<45} {count:>8} {pct:>7.1f}%")

    # ========================================================================
    # 第五部分：LDA 话题分配（补充）
    # ========================================================================
    print("\n" + "─" * 50)
    print("📎 第五部分：LDA 话题分配结果（补充）")
    print("─" * 50)

    print("\n[5.1] 为每篇文档分配 LDA 主导话题...")
    lda_topic_assignments = []
    for bow in corpus:
        topic_probs = best_lda.get_document_topics(bow, minimum_probability=0.0)
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        lda_topic_assignments.append(dominant_topic)

    df_clean['lda_topic_id'] = lda_topic_assignments

    lda_topic_dist = df_clean['lda_topic_id'].value_counts().sort_index()
    print(f"\n  LDA 话题分布统计（K={best_k}）：")
    for topic_id, count in lda_topic_dist.items():
        pct = count / len(df_clean) * 100
        words = best_lda.show_topic(topic_id, topn=5)
        word_str = ', '.join([w for w, _ in words])
        print(f"    话题 {topic_id} ({word_str}): {count} 条 ({pct:.1f}%)")

    # ========================================================================
    # 最终报告
    # ========================================================================
    print("\n" + "=" * 70)
    print("  ✅ 维度二分析完成！")
    print("=" * 70)
    print(f"""
  📁 生成文件清单：
  ├── {os.path.relpath(coherence_fig_path, BASE_DIR)}
  ├── {os.path.relpath(os.path.join(OUTPUT_DIR, 'lda_visualization.html'), BASE_DIR)}
  ├── {os.path.relpath(os.path.join(OUTPUT_DIR, 'bertopic_topics.html'), BASE_DIR)}
  ├── {os.path.relpath(os.path.join(OUTPUT_DIR, 'bertopic_barchart.html'), BASE_DIR)}
  └── {os.path.relpath(OUTPUT_CSV, BASE_DIR)}