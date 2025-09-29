import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

categories = ['54M', '284M', '755M']
values1 = [0.57, 0.59, 0.61]
values2 = [0.69, 0.64, 0.64] 

bar_width = 0.35

bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

plt.bar(bar_positions1, values1, width=bar_width, label='CrossEM', color='moccasin', hatch='//')
plt.bar(bar_positions2, values2, width=bar_width, label='CrossEM+', color='lightblue', hatch='\\')

line_positions = bar_positions1
plt.plot(line_positions, values1, color='black', marker='^', markersize=10, linestyle='-', linewidth=2)
line_positions = bar_positions2
plt.plot(line_positions, values2, color='red', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying data size', fontsize=32)
plt.ylabel('MRR', fontsize=30)

plt.xticks(bar_positions1 + bar_width/2, categories, fontsize=30)
# 设置y轴最大值
plt.ylim(top=1)
plt.yticks(fontsize=30)
plt.subplots_adjust(bottom=0.6, hspace=0.3)


plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('.../exp_mrr.eps', format='eps')

plt.show()
plt.clf()
plt.close('all')







categories = ['54M', '284M', '755M']
values1 = [273.54, 489.72, 812.14]
values2 = [207.71, 235.80, 259.44]

bar_width = 0.35

bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

plt.bar(bar_positions1, values1, width=bar_width, label='CrossEM', color='moccasin', hatch='//')
plt.bar(bar_positions2, values2, width=bar_width, label='CrossEM+', color='lightblue', hatch='\\')

line_positions = bar_positions1
plt.plot(line_positions, values1, color='black', marker='^', markersize=10, linestyle='-', linewidth=2)
line_positions = bar_positions2
plt.plot(line_positions, values2, color='red', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying data size', fontsize=32)
plt.ylabel('Time (Sec)', fontsize=30, labelpad=-10)

plt.xticks(bar_positions1 + bar_width/2, categories, fontsize=30)

plt.ylim(top=1000)
plt.yticks(fontsize=30)
plt.subplots_adjust(bottom=0.6, hspace=0.3)

plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig('.../exp_time.eps', format='eps')

plt.show()
plt.clf()
plt.close('all')




categories = ['54M', '284M', '755M']
values1 = [18.56, 20.98, 23.87]
values2 = [16.11, 18.08, 20.37 ]

bar_width = 0.35

bar_positions1 = np.arange(len(categories))
bar_positions2 = bar_positions1 + bar_width

plt.bar(bar_positions1, values1, width=bar_width, label='CrossEM', color='moccasin', hatch='//')
plt.bar(bar_positions2, values2, width=bar_width, label='CrossEM+', color='lightblue', hatch='\\')

line_positions = bar_positions1
plt.plot(line_positions, values1, color='black', marker='^', markersize=10, linestyle='-', linewidth=2)
line_positions = bar_positions2
plt.plot(line_positions, values2, color='red', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying data size', fontsize=32)
plt.ylabel('Memory (GB)', fontsize=30)

plt.xticks(bar_positions1 + bar_width/2, categories, fontsize=30)

plt.ylim(top=30)
plt.yticks(fontsize=30)
plt.subplots_adjust(bottom=0.6, hspace=0.3)

plt.legend(fontsize=20)
plt.tight_layout()

plt.savefig('.../exp_memory.eps', format='eps')

plt.show()
plt.clf()
plt.close('all')





x = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
y = [0.83, 0.84, 0.86, 0.85, 0.84, 0.83]

plt.plot(x, y, color='blue', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying theta', fontsize=30)
plt.ylabel('MRR', fontsize=30)

plt.xticks(x[::2], fontsize=30)

plt.yticks([0.6, 0.7, 0.8, 0.9, 1], fontsize=30)

plt.subplots_adjust(bottom=0.6, hspace=0.3)

plt.tight_layout()

plt.savefig('.../exp_theta_mrr.eps', format='eps')

plt.clf()
plt.close('all')





x = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
y = [65.28, 48.14, 25.79, 10.80, 4.49, 1.19]

plt.plot(x, y, color='blue', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying theta', fontsize=32)
plt.ylabel('Time (Sec)', fontsize=30)

plt.xticks(x[::2], fontsize=30)
plt.yticks([0, 20, 40, 60, 80], fontsize=30)

plt.subplots_adjust(bottom=0.6, hspace=0.3)

plt.tight_layout()
plt.savefig('.../exp_theta_time.eps', format='eps')

plt.clf()
plt.close('all')





x = [0, 0.005, 0.01, 0.015, 0.02, 0.025]
y = [10.37, 9.77, 9.36, 10.35, 11.06, 11.49 ]

plt.plot(x, y, color='blue', marker='^', markersize=10, linestyle='-', linewidth=2)

plt.xlabel('Varying theta', fontsize=32)
plt.ylabel('Memory (GB)', fontsize=30)

plt.xticks(x[::2], fontsize=30)
plt.yticks([8, 9, 10, 11, 12], fontsize=30)

plt.subplots_adjust(bottom=0.6, hspace=0.3)

plt.tight_layout()
plt.savefig('.../exp_theta_memory.eps', format='eps')

plt.clf()
plt.close('all')
