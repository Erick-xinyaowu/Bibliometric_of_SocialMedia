import pandas as pd
from datetime import datetime

# 读取两个CSV文件
print("正在读取文件...")
df_comments = pd.read_csv('data/douyin_data_with_topics.csv')
df_videos = pd.read_csv('data/search_contents_2026-04-15_formatted.csv')

print(f"评论数据记录数: {len(df_comments)}")
print(f"视频数据记录数: {len(df_videos)}")

# 转换为日期格式
df_comments['create_time'] = pd.to_datetime(df_comments['create_time'], errors='coerce')
df_videos['create_time'] = pd.to_datetime(df_videos['create_time'], errors='coerce')

# 基于aweme_id进行左连接
print("\n正在基于aweme_id进行匹配...")
df_merged = df_comments.merge(
    df_videos[['aweme_id', 'create_time']],
    on='aweme_id',
    how='left',
    suffixes=('', '_video')
)

# 新增字段1: base_time (视频发布时间)
df_merged['base_time'] = df_merged['create_time_video']

# 新增字段2: delta_T (时间差，以天为单位)
df_merged['delta_T'] = (df_merged['create_time'] - df_merged['create_time_video']).dt.days

# 将日期格式化回 YYYY-MM-DD
df_merged['create_time'] = df_merged['create_time'].dt.strftime('%Y-%m-%d')
df_merged['base_time'] = df_merged['base_time'].dt.strftime('%Y-%m-%d')

# 删除临时列 create_time_video
df_merged = df_merged.drop(columns=['create_time_video'])

# 统计匹配情况
matched_count = df_merged['base_time'].notna().sum()
unmatched_count = df_merged['base_time'].isna().sum()

print(f"\n匹配统计:")
print(f"  - 成功匹配: {matched_count} 条")
print(f"  - 未匹配: {unmatched_count} 条")
print(f"  - 匹配率: {matched_count/len(df_merged)*100:.2f}%")

# 保存到新文件
output_file = 'data/douyin_data_with_topics_enriched.csv'
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n处理完成! 输出文件: {output_file}")
print(f"输出记录数: {len(df_merged)}")
print(f"输出字段数: {len(df_merged.columns)}")

# 显示新增字段的统计信息
print("\ndelta_T 统计:")
print(df_merged['delta_T'].describe())

# 显示前5条示例数据
print("\n示例数据 (前5条):")
print(df_merged[['aweme_id', 'create_time', 'base_time', 'delta_T']].head())
