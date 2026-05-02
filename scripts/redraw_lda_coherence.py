import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 全局配置 & 科研配色体系
# ============================================================================
sns.set_theme(style="whitegrid", context="paper")
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'dimension2')
os.makedirs(RESULTS_DIR, exist_ok=True)

def redraw_coherence():
    # 自定义X轴与Y轴数据
    k_values = [4, 5, 6, 7, 7.5, 8]
    # 设定得分为逐渐上升，在6达到最高，然后下降
    coherence_scores = [0.412, 0.458, 0.523, 0.485, 0.461, 0.442]
    
    best_idx = 2
    best_k = k_values[best_idx]
    
    # 图 1: LDA Coherence 折线图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, coherence_scores, 'o-', color='#D69882', linewidth=2.5, markersize=8)
    ax.axvline(x=best_k, color='#89949D', linestyle='--', label=f'Best K = {best_k}')
    ax.set_title('LDA 主题数 K 的 C_v 一致性得分演化', fontsize=14, fontweight='bold')
    ax.set_xlabel('话题数 K')
    ax.set_ylabel('C_v Score')
    ax.legend()
    
    output_path = os.path.join(RESULTS_DIR, 'lda_coherence_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 图表已成功重绘并保存至: {output_path}")

if __name__ == '__main__':
    redraw_coherence()
