import matplotlib.pyplot as plt
import numpy as np

# 数据准备
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
metrics = {
    'RMOTA': [-31.98, -9.44, 0.30, 5.89, 9.14],
    'RMOTP': [84.40, 84.97, 85.70, 87.06, 87.82],
    'Precision': [35.80, 45.51, 51.79, 58.69, 69.38],
    'Recall': [38.27, 37.06, 33.81, 25.38, 18.17]
}

# 创建双轴图表
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

# 左轴曲线（MOTA/MOTP）
ax1.plot(thresholds, metrics['RMOTA'], 'b-o', label='RMOTA (%)', markersize=8, linewidth=2)
ax1.plot(thresholds, metrics['RMOTP'], 'g--s', label='RMOTP (%)', markersize=8, linewidth=2)

# 右轴曲线（Precision/Recall）
ax2.plot(thresholds, metrics['Precision'], 'r-^', label='Precision (%)', markersize=8, linewidth=2)
ax2.plot(thresholds, metrics['Recall'], 'm:v', label='Recall (%)', markersize=8, linewidth=2)

# 图表装饰
ax1.set_xlabel('Detection Threshold', fontsize=12)
ax1.set_ylabel('RMOTA/RMOTP (%)', color='k', fontsize=12)
ax2.set_ylabel('Precision/Recall (%)', color='k', fontsize=12)
ax1.tick_params(axis='y', labelcolor='k')
ax2.tick_params(axis='y', labelcolor='k')
plt.title('Tracking Performance Metrics', fontsize=16, pad=10)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))

plt.tight_layout()
plt.savefig('Tracking_Performance_Metrics.eps', format='eps', dpi=600)
plt.show()