# -*- coding: utf-8 -*-
"""维度四诊断脚本：彻底摸清数据底牌"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

df = pd.read_csv('data/douyin_data_with_topics_enriched.csv')
df = df[df['delta_T'] >= 0].copy()

print("=== 1. 数据集概况 ===")
print(f"总评论数: {len(df)}")
print(f"唯一短视频数: {df['aweme_id'].nunique()}")

print("\n=== 2. 各视频评论数分布 ===")
vc = df.groupby('aweme_id').size()
print(vc.describe())

print("\n=== 3. sub_comment_count 分布 ===")
print(df['sub_comment_count'].describe())
print(f"非零二级评论占比: {(df['sub_comment_count']>0).mean()*100:.1f}%")

print("\n=== 4. 按视频聚合 ===")
va = df.groupby('aweme_id').agg(
    lifespan=('delta_T', 'max'),
    avg_sub=('sub_comment_count', 'mean'),
    total_sub=('sub_comment_count', 'sum'),
    avg_like=('like_count', 'mean'),
    n_comments=('comment_id', 'count')
).reset_index()
print(va.describe())

print("\n=== 5. 视频级 Spearman 相关矩阵 (全部视频) ===")
for c1 in ['avg_sub','total_sub','avg_like']:
    rho, p = spearmanr(va[c1], va['lifespan'])
    print(f"  {c1:12s} vs lifespan: rho={rho:.4f}, p={p:.4f}")

print("\n=== 6. 过滤 total_comments>3 后 ===")
va2 = va[va['n_comments'] > 3]
print(f"剩余视频数: {len(va2)}")
for c1 in ['avg_sub','total_sub','avg_like']:
    rho, p = spearmanr(va2[c1], va2['lifespan'])
    print(f"  {c1:12s} vs lifespan: rho={rho:.4f}, p={p:.4f}")

print("\n=== 7. 话题级聚合 (排除离群点) ===")
df_valid = df[~df['topic_label'].str.contains('离群点', na=True)]
ta = df_valid.groupby('topic_label').agg(
    avg_sub=('sub_comment_count', 'mean'),
    total_sub=('sub_comment_count', 'sum'),
    avg_like=('like_count', 'mean'),
    n_comments=('comment_id', 'count'),
    lifespan=('delta_T', 'max')
).reset_index()

aging = pd.read_csv('results/dimension3/topic_aging_metrics.csv')
tm = ta.merge(aging, left_on='topic_label', right_on='Topic', how='inner')
print(f"话题数: {len(tm)}")
print(tm[['topic_label','avg_sub','total_sub','avg_like','T_1/2 (Days)','lifespan']].to_string())

print("\n=== 8. 话题级 Spearman ===")
for c1 in ['avg_sub','total_sub','avg_like','n_comments']:
    rho, p = spearmanr(tm[c1], tm['T_1/2 (Days)'])
    print(f"  {c1:12s} vs T_1/2: rho={rho:.4f}, p={p:.4f}")
    rho2, p2 = spearmanr(tm[c1], tm['lifespan'])
    print(f"  {c1:12s} vs lifespan: rho={rho2:.4f}, p={p2:.4f}")
