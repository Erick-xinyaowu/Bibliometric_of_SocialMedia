# -*- coding: utf-8 -*-
"""
=============================================================================
 维度四：演化关联分析 — 重构版（尊重数据的诚实学术路线）
=============================================================================
基于数据诊断结果重构：
  - 图1: 话题级多变量 Spearman 关联热力图（全景矩阵）
  - 图2: 评论总量 vs 话题存续期（显著正相关，P=0.035）
  - 图3: 扩散深度 vs 半衰期（诚实展示不显著结果，P=0.69）
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings

# ============================================================================
# 全局配置
# ============================================================================
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DIM3_RESULTS = os.path.join(BASE_DIR, 'results', 'dimension3')
DIM4_RESULTS = os.path.join(BASE_DIR, 'results', 'dimension4')
os.makedirs(DIM4_RESULTS, exist_ok=True)

COMMENTS_FILE = os.path.join(DATA_DIR, 'douyin_data_with_topics_enriched.csv')
AGING_FILE = os.path.join(DIM3_RESULTS, 'topic_aging_metrics.csv')

def get_asterisks(p_val):
    if p_val < 0.001: return "***"
    elif p_val < 0.01: return "**"
    elif p_val < 0.05: return "*"
    else: return "ns"


def main():
    print("=" * 70)
    print("  维度四：演化关联分析 — 重构版（诚实学术路线）")
    print("=" * 70)

    # ================================================================
    # 1. 数据融合：话题级聚合 + 维度三半衰期 Join
    # ================================================================
    print("\n[1] 数据融合与话题级聚合...")
    df = pd.read_csv(COMMENTS_FILE)
    df = df[df['delta_T'] >= 0].copy()
    df_valid = df[~df['topic_label'].str.contains('离群点', na=True)].copy()

    topic_agg = df_valid.groupby('topic_label').agg(
        avg_like=('like_count', 'mean'),
        avg_sub=('sub_comment_count', 'mean'),
        total_sub=('sub_comment_count', 'sum'),
        n_comments=('comment_id', 'count'),
        lifespan=('delta_T', 'max')
    ).reset_index()

    aging = pd.read_csv(AGING_FILE)
    tm = topic_agg.merge(aging, left_on='topic_label', right_on='Topic', how='inner')
    N = len(tm)
    print(f"  ok 成功融合 {N} 个语义话题的跨维度指标。")

    # ================================================================
    # 2. 图1: 话题级多变量 Spearman 全景热力图
    # ================================================================
    print("\n[2] 绘制话题级多变量 Spearman 全景热力图...")
    vars_to_test = ['avg_like', 'avg_sub', 'n_comments', 'T_1/2 (Days)', 'lifespan']
    labels = [
        '扩散强度\n(平均点赞)',
        '扩散深度\n(平均二级评论)',
        '扩散广度\n(评论总量)',
        '半衰期\n($T_{1/2}$)',
        '话题存续期\n(最大 $\\Delta T$)'
    ]
    n_vars = len(vars_to_test)
    corr_matrix = np.zeros((n_vars, n_vars))
    annot_labels = np.empty((n_vars, n_vars), dtype=object)
    stats_records = []

    for i in range(n_vars):
        for j in range(n_vars):
            rho, pval = spearmanr(tm[vars_to_test[i]], tm[vars_to_test[j]])
            corr_matrix[i, j] = rho
            sig = get_asterisks(pval)
            annot_labels[i, j] = f"{rho:.2f}\n({sig}, p={pval:.3f})"
            if i < j:
                stats_records.append({
                    'Variable_1': labels[i].replace('\n', ' '),
                    'Variable_2': labels[j].replace('\n', ' '),
                    'Spearman_Rho': round(rho, 4),
                    'P_Value': round(pval, 6),
                    'Significance': sig
                })

    pd.DataFrame(stats_records).to_csv(
        os.path.join(DIM4_RESULTS, 'dimension4_spearman_stats.csv'),
        index=False, encoding='utf-8-sig'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=annot_labels, fmt="",
        cmap='RdBu_r', vmin=-1, vmax=1, center=0,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Spearman $\\rho$'},
        annot_kws={'size': 10, 'weight': 'bold'},
        linewidths=0.5, linecolor='white'
    )
    plt.title(
        f'话题级多维扩散特征与老化指标 Spearman 关联全景矩阵 (N={N})',
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xticks(fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(DIM4_RESULTS, 'spearman_heatmap_full.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ok 全景热力图已保存。")

    # ================================================================
    # 3. 图2: 评论总量 vs 话题存续期（显著正相关 P=0.035）
    # ================================================================
    print("[3] 绘制评论总量 vs 话题存续期散点图（显著正相关）...")
    rho_sig, p_sig = spearmanr(tm['n_comments'], tm['lifespan'])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(tm['n_comments'], tm['lifespan'],
               s=100, c='#2E5B50', alpha=0.8, edgecolors='white', zorder=3)

    # 为每个话题添加简短标签
    for _, row in tm.iterrows():
        short_name = row['topic_label'].replace('Topic', 'T').split('_')[0] + '_' + '_'.join(row['topic_label'].split('_')[1:3])
        ax.annotate(short_name, (row['n_comments'], row['lifespan']),
                    fontsize=7, alpha=0.7, ha='left',
                    xytext=(5, 3), textcoords='offset points')

    # 添加 OLS 趋势线
    z = np.polyfit(tm['n_comments'], tm['lifespan'], 1)
    p_line = np.poly1d(z)
    x_smooth = np.linspace(tm['n_comments'].min(), tm['n_comments'].max(), 100)
    ax.plot(x_smooth, p_line(x_smooth), color='#A63A28', linewidth=2.5, linestyle='--', zorder=2)

    ax.set_xlabel('扩散广度（话题下评论总量）', fontsize=12, fontweight='bold')
    ax.set_ylabel('话题存续期（最大 $\\Delta T$，天）', fontsize=12, fontweight='bold')
    ax.set_title('扩散广度与话题存续寿命的正向关联验证', fontsize=14, fontweight='bold', pad=15)

    textstr = f'Spearman $\\rho$ = {rho_sig:.3f}{get_asterisks(p_sig)}\np = {p_sig:.4f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(DIM4_RESULTS, 'breadth_vs_lifespan.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ok 显著正相关图已保存 (rho={rho_sig:.4f}, p={p_sig:.4f})。")

    # ================================================================
    # 4. 图3: 扩散深度 vs 半衰期（诚实展示不显著结果）
    # ================================================================
    print("[4] 绘制扩散深度 vs 半衰期散点图（诚实展示不显著）...")
    rho_ns, p_ns = spearmanr(tm['avg_sub'], tm['T_1/2 (Days)'])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(tm['avg_sub'], tm['T_1/2 (Days)'],
               s=100, c='#89949D', alpha=0.8, edgecolors='white', zorder=3)

    for _, row in tm.iterrows():
        short_name = row['topic_label'].replace('Topic', 'T').split('_')[0] + '_' + '_'.join(row['topic_label'].split('_')[1:3])
        ax.annotate(short_name, (row['avg_sub'], row['T_1/2 (Days)']),
                    fontsize=7, alpha=0.7, ha='left',
                    xytext=(5, 3), textcoords='offset points')

    # 灰色趋势线（表明无显著趋势）
    z2 = np.polyfit(tm['avg_sub'], tm['T_1/2 (Days)'], 1)
    p_line2 = np.poly1d(z2)
    x_smooth2 = np.linspace(tm['avg_sub'].min(), tm['avg_sub'].max(), 100)
    ax.plot(x_smooth2, p_line2(x_smooth2), color='#B6B0B9', linewidth=2, linestyle=':', zorder=2)

    ax.set_xlabel('扩散深度（话题下平均二级评论数）', fontsize=12, fontweight='bold')
    ax.set_ylabel('注意力半衰期 $T_{1/2}$（天）', fontsize=12, fontweight='bold')
    ax.set_title('扩散深度与注意力半衰期的关联检验（不显著）', fontsize=14, fontweight='bold', pad=15)

    textstr2 = f'Spearman $\\rho$ = {rho_ns:.3f} ({get_asterisks(p_ns)})\np = {p_ns:.4f}'
    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#FFF3F3', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(DIM4_RESULTS, 'depth_vs_halflife.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ok 不显著结果图已保存 (rho={rho_ns:.4f}, p={p_ns:.4f})。")

    # ================================================================
    # 5. 图4: 扩散强度 vs 扩散深度（话题级，极显著 P=0.007）
    # ================================================================
    print("[5] 绘制扩散强度 vs 扩散深度话题级散点图（极显著协同效应）...")
    rho_syn, p_syn = spearmanr(tm['avg_like'], tm['avg_sub'])

    fig, ax = plt.subplots(figsize=(9, 6))

    # 气泡大小：用 sqrt 拉大差距，最小 200，最大 4000
    norm_sizes = np.sqrt(tm['n_comments'])
    sizes = 200 + (norm_sizes - norm_sizes.min()) / (norm_sizes.max() - norm_sizes.min()) * 3800
    scatter = ax.scatter(tm['avg_like'], tm['avg_sub'],
                         s=sizes, c='#2E5B50', alpha=0.7, edgecolors='white', linewidths=1.5, zorder=3)

    # 对数坐标消除离群点视觉挤压
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 直线拟合（对数空间中的 OLS 直线）
    log_x = np.log10(tm['avg_like'].clip(lower=0.1)).values
    log_y = np.log10(tm['avg_sub'].clip(lower=0.1)).values
    z3 = np.polyfit(log_x, log_y, 1)
    p_line3 = np.poly1d(z3)
    x_s3 = np.linspace(log_x.min(), log_x.max(), 100)
    ax.plot(10**x_s3, 10**p_line3(x_s3), color='#A63A28', linewidth=2.5, linestyle='--', zorder=2)

    ax.set_xlabel('扩散强度（话题下平均点赞数，对数轴）', fontsize=12, fontweight='bold')
    ax.set_ylabel('扩散深度（话题下平均二级评论数，对数轴）', fontsize=12, fontweight='bold')
    ax.set_title('扩散强度与深度的协同效应验证（话题级，Log-Log）', fontsize=14, fontweight='bold', pad=15)

    textstr3 = f'Spearman $\\rho$ = {rho_syn:.3f}{get_asterisks(p_syn)}\np = {p_syn:.4f}'
    ax.text(0.05, 0.95, textstr3, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # 图例说明气泡大小
    ax.text(0.95, 0.05, '气泡大小 = 评论总量\n(扩散广度)', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom', style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(DIM4_RESULTS, 'synergy_likes_vs_depth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ok 协同效应散点图已保存 (rho={rho_syn:.4f}, p={p_syn:.4f})。")


    # ================================================================
    # 6. 图5: 评论级分层——点赞量分位数组的平均二级评论率阶梯图
    # ================================================================
    print("[6] 绘制评论级点赞量分层的二级评论率阶梯图...")
    df_all = pd.read_csv(COMMENTS_FILE)
    df_all = df_all[df_all['delta_T'] >= 0].copy()

    # 手动定义符合人类直觉的点赞量区间
    bins = [0, 1, 2, 4, 11, 51, float('inf')]
    bin_labels = ['0 赞', '1 赞', '2-3 赞', '4-10 赞', '11-50 赞', '50+ 赞']
    df_all['like_tier'] = pd.cut(df_all['like_count'], bins=bins, labels=bin_labels,
                                  right=False, include_lowest=True)

    # 剔除二级评论数的极端离群点（>99 分位），防止柱高被少数极端值拉飞
    cap = df_all['sub_comment_count'].quantile(0.99)
    df_capped = df_all[df_all['sub_comment_count'] <= cap].copy()

    tier_stats = df_capped.groupby('like_tier', observed=False).agg(
        avg_sub=('sub_comment_count', 'mean'),
        has_sub_ratio=('sub_comment_count', lambda x: (x > 0).mean() * 100),
        n=('comment_id', 'count')
    ).reset_index()
    tier_stats.columns = ['label', 'avg_sub', 'has_sub_ratio', 'n']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x_pos = range(len(tier_stats))
    bars = ax1.bar(x_pos, tier_stats['avg_sub'], color='#2E5B50', alpha=0.85, width=0.5, zorder=3)

    # 在每个柱子上方标注数值
    for i, (val, ratio) in enumerate(zip(tier_stats['avg_sub'], tier_stats['has_sub_ratio'])):
        ax1.text(i, val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 叠加折线：有二级评论的比例
    ax2 = ax1.twinx()
    ax2.plot(x_pos, tier_stats['has_sub_ratio'], color='#A63A28', marker='o',
             linewidth=2.5, markersize=8, zorder=4)
    ax2.set_ylabel('产生二级评论的概率 (%)', fontsize=11, fontweight='bold', color='#A63A28')
    ax2.tick_params(axis='y', labelcolor='#A63A28')

    for i, ratio in enumerate(tier_stats['has_sub_ratio']):
        ax2.annotate(f'{ratio:.1f}%', (i, ratio), fontsize=9, color='#A63A28',
                     fontweight='bold', ha='center',
                     xytext=(0, 10), textcoords='offset points')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tier_stats['label'], fontsize=11)
    ax1.set_xlabel('评论点赞量分层 (由低到高)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('平均二级评论数', fontsize=12, fontweight='bold')
    ax1.set_title('注意力强度分层下的深度互动激发效应（评论级 N={:,}，剔除极端离群点）'.format(len(df_capped)),
                  fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(DIM4_RESULTS, 'like_tier_sub_comment_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ok 分层阶梯图已保存。")

    # ================================================================
    # 7. 汇总输出
    # ================================================================
    print("\n" + "=" * 70)
    print("  维度四重构执行完毕！关键发现摘要：")
    print(f"  [显著]   扩散广度 vs 存续期:   rho={rho_sig:.4f}, p={p_sig:.4f} *")
    print(f"  [极显著] 强度 vs 深度协同效应: rho={rho_syn:.4f}, p={p_syn:.4f} **")
    print(f"  [不显著] 扩散深度 vs 半衰期:   rho={rho_ns:.4f}, p={p_ns:.4f} ns")
    print("=" * 70)


if __name__ == "__main__":
    main()

