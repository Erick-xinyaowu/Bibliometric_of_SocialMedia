# -*- coding: utf-8 -*-
"""
=============================================================================
 维度三：注意力老化测度 — 多模型衰减对抗拟合与半衰期 (T_1/2) 测算
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings

# ============================================================================
# 全局配置 & 科研配色体系（已优化）
# ============================================================================
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ 科研标准：色盲友好（Okabe-Ito 配色）
COLOR_PALETTE = [
    "#0072B2",  # 蓝
    "#D55E00",  # 橙
    "#009E73",  # 绿
    "#CC79A7",  # 紫
    "#F0E442",  # 黄
    "#56B4E9",  # 浅蓝
    "#E69F00",  # 金橙
    "#000000"   # 黑
]
sns.set_palette(COLOR_PALETTE)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'douyin_data_with_topics_enriched.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'dimension3')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 拟合用的数学衰减方程
def exp_decay(x, A, lam):
    """指数衰减模型: y = A * e^(-lam * x)"""
    return A * np.exp(-lam * x)

def power_law(x, A, alpha):
    """幂律衰减模型: y = A * x^(-alpha)"""
    return A * np.power(x, -alpha)

def lognormal_decay(x, A, mu, sigma):
    """对数正态衰减模型: 刻画爆发-衰减的全周期"""
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (A / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))
    return np.nan_to_num(val)

# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 70)
    print("  维度三：注意力老化测度 — 多模型竞争拟合与生命周期评估")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载已打标的融合数据集...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=['delta_T', 'like_count'])
    
    df = df[df['delta_T'] >= 0].copy()
    df['aging_days'] = df['delta_T'] + 1 

    # 2. 宏观衰减：多尺度分箱拟合
    print("\n[2] 执行多尺度分箱的全局注意力衰减拟合测试...")
    bins_list = [1, 3, 7]
    fit_results = []

    for b in bins_list:
        print(f"  → 正在处理分箱粒度: {b} 天...")
        
        df[f'bin_{b}d'] = ((df['aging_days'] - 1) // b) * b + 1
        grouped = df.groupby(f'bin_{b}d')['like_count'].sum().reset_index()
        grouped = grouped.sort_values(by=f'bin_{b}d')
        
        x_data = grouped[f'bin_{b}d'].values
        y_data = grouped['like_count'].values / b 
        
        mask = y_data > 0
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        
        if len(x_fit) < 4:
            continue
            
        p0_exp = [max(y_fit), 0.1]
        p0_pow = [max(y_fit), 1.0]
        p0_log = [sum(y_fit)*b, np.log(np.mean(x_fit)), 1.0]

        best_r2 = -1
        best_model_name = ""
        models_metrics = {}
        
        plt.figure(figsize=(10, 6))
        
        # ✅ 优化：数据点弱化
        plt.scatter(
            x_fit, y_fit,
            color='#B0B0B0',
            s=60,
            alpha=0.7,
            edgecolors='none',
            label='Empirical Density (实证点赞均值)'
        )

        # (a) 指数衰减
        try:
            popt_exp, _ = curve_fit(exp_decay, x_fit, y_fit, p0=p0_exp, maxfev=10000)
            y_pred_exp = exp_decay(x_fit, *popt_exp)
            r2_exp = r2_score(y_fit, y_pred_exp)
            models_metrics['Exponential'] = r2_exp
            plt.plot(x_fit, y_pred_exp, color='#0072B2', linewidth=2.5, linestyle='--',
                     label=f'Exp Decay ($R^2$={r2_exp:.4f})')
            if r2_exp > best_r2: best_r2, best_model_name = r2_exp, 'Exponential'
        except:
            pass

        # (b) 幂律衰减
        try:
            popt_pow, _ = curve_fit(power_law, x_fit, y_fit, p0=p0_pow, maxfev=10000)
            y_pred_pow = power_law(x_fit, *popt_pow)
            r2_pow = r2_score(y_fit, y_pred_pow)
            models_metrics['Power-Law'] = r2_pow
            plt.plot(x_fit, y_pred_pow, color='#D55E00', linewidth=2.5, linestyle='-.',
                     label=f'Power-Law ($R^2$={r2_pow:.4f})')
            if r2_pow > best_r2: best_r2, best_model_name = r2_pow, 'Power-Law'
        except:
            pass

        # (c) 对数正态
        try:
            bounds = ([0, -np.inf, 0.01], [np.inf, np.inf, 10.0])
            popt_log, _ = curve_fit(lognormal_decay, x_fit, y_fit, p0=p0_log, bounds=bounds, maxfev=10000)
            y_pred_log = lognormal_decay(x_fit, *popt_log)
            r2_log = r2_score(y_fit, y_pred_log)
            models_metrics['Lognormal'] = r2_log
            plt.plot(x_fit, y_pred_log, color='#009E73', linewidth=3,
                     label=f'Lognormal ($R^2$={r2_log:.4f})')
            if r2_log > best_r2: best_r2, best_model_name = r2_log, 'Lognormal'
        except Exception as e:
            pass

        plt.title(f'全局信息老化过程多模型对抗拟合 (分箱粒度: $\Delta T$={b} 天)', fontsize=14, fontweight='bold')
        plt.xlabel('视频发布后的时间 (aging_days)')
        plt.ylabel('日均新增点赞规模 (Volume/Day)')
        plt.legend(frameon=True, shadow=True, title="竞争拟合模型")
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f'macro_decay_fit_bin{b}d.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        fit_results.append({
            'Bin_Size': f"{b} Days",
            'Best_Model': best_model_name,
            'Best_R2': best_r2,
            'Metrics': models_metrics
        })

    print("\n  📊 多模型竞争拟合度 (R²) 结果汇报:")
    for res in fit_results:
        print(f"    - 分箱 {res['Bin_Size']:<7} | 最佳模型: {res['Best_Model']:<12} (R² = {res['Best_R2']:.4f})")
        print(f"      [各模型详情]: {', '.join([f'{k}={v:.4f}' for k, v in res['Metrics'].items()])}")

    # 3. 半衰期
    print("\n[3] 测算各深度子话题的半衰期 (T_1/2)...")
    valid_df = df[(df['topic_label'].notna()) & (~df['topic_label'].str.contains('离群点'))].copy()
    
    half_life_records = []
    
    for topic in valid_df['topic_label'].unique():
        topic_df = valid_df[valid_df['topic_label'] == topic].copy()
        
        topic_grouped = topic_df.groupby('aging_days')['like_count'].sum().reset_index().sort_values('aging_days')
        total_likes = topic_grouped['like_count'].sum()
        
        if total_likes < 100:
            continue
            
        topic_grouped['cum_likes'] = topic_grouped['like_count'].cumsum()
        topic_grouped['cum_ratio'] = topic_grouped['cum_likes'] / total_likes
        
        half_life_row = topic_grouped[topic_grouped['cum_ratio'] >= 0.5]
        t_half = half_life_row.iloc[0]['aging_days'] if not half_life_row.empty else topic_grouped['aging_days'].max()
        
        t_90_row = topic_grouped[topic_grouped['cum_ratio'] >= 0.9]
        t_90 = t_90_row.iloc[0]['aging_days'] if not t_90_row.empty else topic_grouped['aging_days'].max()
        
        half_life_records.append({
            'Topic': topic,
            'Total_Attention': total_likes,
            'T_1/2 (Days)': t_half,
            'T_90% (Days)': t_90
        })
        
    hl_df = pd.DataFrame(half_life_records).sort_values(by='T_1/2 (Days)', ascending=True)
    hl_df.to_csv(os.path.join(RESULTS_DIR, 'topic_aging_metrics.csv'), index=False, encoding='utf-8-sig')
    
    print(f"  ✓ 已完成 {len(hl_df)} 个核心子话题的寿命测算。")

    # 4. 半衰期图（单色科研风）
    plt.figure(figsize=(10, 6))
    sns.barplot(data=hl_df, y='Topic', x='T_1/2 (Days)', color='#4C72B0')
    plt.title('各生成式 AI 子话题注意力半衰期 ($T_{1/2}$) 阶梯图', fontweight='bold', fontsize=14)
    plt.xlabel('达到 50% 总点赞量所需的天数 (衰减极值)')
    plt.ylabel('')
    for i, v in enumerate(hl_df['T_1/2 (Days)']):
        plt.text(v + 0.1, i, str(v), color='black', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topic_half_life_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 生命周期曲线
    print("  → 绘制典型子话题生命周期累计分布曲线...")
    top4_topics = hl_df.sort_values(by='Total_Attention', ascending=False).head(4)['Topic'].tolist()
    
    plt.figure(figsize=(10, 6))
    
    # ✅ 高区分科研配色
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    
    for idx, topic in enumerate(top4_topics):
        topic_df = valid_df[valid_df['topic_label'] == topic].copy()
        topic_grouped = topic_df.groupby('aging_days')['like_count'].sum().reset_index().sort_values('aging_days')
        topic_grouped['cum_ratio'] = topic_grouped['like_count'].cumsum() / topic_grouped['like_count'].sum()
        
        topic_grouped = topic_grouped[topic_grouped['aging_days'] <= 60]
        plt.plot(topic_grouped['aging_days'], topic_grouped['cum_ratio'], 
                 linewidth=2.5, color=colors[idx], label=topic)
                 
    plt.axhline(y=0.5, color='#666666', linestyle='--', alpha=0.7)
    plt.text(60, 0.51, 'Half-Life Threshold (50%)', ha='right', color='#666666')
    plt.title('典型高频子话题注意力累积爆发与衰老轨迹 (CDF)', fontweight='bold', fontsize=14)
    plt.xlabel('视频发布后的天数 ($\Delta T$)')
    plt.ylabel('累计获取的注意力占比 (Cumulative Ratio)')
    plt.legend(title="Top 4 热点话题")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topic_lifecycle_compare.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ 维度三计算已圆满完成！所有图表及排行榜已经存入 results/dimension3/ 目录。")

if __name__ == "__main__":
    main()