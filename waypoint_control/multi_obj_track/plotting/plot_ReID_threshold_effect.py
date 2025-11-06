import matplotlib.pyplot as plt
import numpy as np

# 数据准备
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
MPE = [45.8713, 50.8541, 30.5668, 32.2842, 34.4164]
MaxPE = [85.9068, 98.2202, 65.0764, 55.1537, 52.3780]
MFPE = [51.0212, 50.8812, 37.8725, 39.9763, 33.9278]
TOR = [41.05, 50.68, 42.69, 36.27, 29.53]

# 创建图形
fig, ax1 = plt.subplots(figsize=(8, 6))  # 接近正方形，略宽
ax2 = ax1.twinx()

# 左轴：误差指标
ax1.plot(thresholds, MPE, 'b-o', label='MPE (m)', linewidth=2, markersize=8)
ax1.plot(thresholds, MaxPE, 'orange', linestyle='--', marker='s', label='MaxPE (m)', linewidth=2, markersize=8)
ax1.plot(thresholds, MFPE, 'g-^', label='MFPE (m)', linewidth=2, markersize=8)

# 右轴：重合度（TOR %）
ax2.plot(thresholds, TOR, 'r:v', label='TOR (%)', linewidth=2, markersize=8)

# 坐标轴标签
ax1.set_xlabel('Re-identification Threshold', fontsize=12)
ax1.set_ylabel('Error Metrics (m)', fontsize=12)
ax2.set_ylabel('Overlap Ratio (%)', fontsize=12)

# 坐标轴刻度
ax1.tick_params(axis='both', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)

# 标题
plt.title('Trajectory Performance Metrics', fontsize=16, pad=10)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15), fontsize=11)

# 网格
# ax1.grid(False, linestyle='--', alpha=0.5)

plt.tight_layout()

# 保存为 EPS 矢量图
plt.savefig('Trajectory_Performance_Metrics.eps', format='eps', dpi=600)

plt.show()
