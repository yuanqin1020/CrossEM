
import matplotlib.pyplot as plt
import numpy as np

x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
y1 = [0.81, 0.83, 0.86, 0.85, 0.84, 0.83]
y2 = [102, 93, 42, 73, 56, 49]
y3 = [10, 10.4, 9.33, 9.86, 9.72, 9.65]

# 创建三个子图
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 8)) 

# subplot1
axes1 = axes[0]
axes1.plot(x, y1, color='#1f77b4', ls='-', marker='^')
axes1.set_xticks(x)
axes1.set_yticks(np.arange(0.7, 1, 0.1))
axes1.set_ylabel('MRR', fontsize=30)
axes1.set_ylim(0.69, 1.01)

# subplot2
axes2 = axes[1]
axes2.plot(x, y2, color='#ff7f0e', ls='-', marker='^')
axes1.set_xticks(x)
axes2.set_yticks(np.arange(0, 121, 40))
axes2.set_ylabel('Time', fontsize=30)
axes2.set_ylim(-1, 121)

# subplot3
axes3 = axes[2]
axes3.plot(x, y3, color='blue', ls='-', marker='^')
axes1.set_xticks(x)
axes3.set_yticks(np.arange(6, 13, 2))
axes3.set_ylabel('Memory', fontsize=30)
axes3.set_ylim(5.8, 12.2)

# 设置所有xlabel, ylable, xticks, yticks的字体大小为20
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=30)

# 展示图片
plt.tight_layout()

# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Prompt Tuning EM/submission in revision/figures/exp_cub_alpha.eps', format='eps')
plt.clf()
plt.close('all')



x = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
y1 = [0.55, 0.56, 0.57, 0.56, 0.56, 0.55] 
y2 = [545, 161, 118, 258, 458, 546]
y3 = [9.28, 9.75, 10.2, 10.35, 10.31, 10.10]

# 创建三个子图
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8,8)) 

# subplot1
axes1 = axes[0]
axes1.plot(x, y1, color='#1f77b4', ls='-', marker='^')
axes1.set_yticks(np.arange(0.4, 0.6, 0.1))
axes1.set_xticks(x)
axes1.set_ylabel('MRR', fontsize=30)
axes1.set_ylim(0.39, 0.61)

# subplot2
axes2 = axes[1]
axes2.plot(x, y2, color='#ff7f0e', ls='-', marker='^')
axes1.set_xticks(x)
axes2.set_yticks(np.arange(100, 600, 150))
axes2.set_ylabel('Time', fontsize=30)
axes2.set_ylim(98, 602)

# subplot3
axes3 = axes[2]
axes3.plot(x, y3, color='blue', ls='-', marker='^')
axes1.set_xticks(x)
axes3.set_yticks(np.arange(9, 11, 1))
axes3.set_ylabel('Memory', fontsize=30)
axes3.set_ylim(8.8, 11.2)

# 设置所有xlabel, ylable, xticks, yticks的字体大小为20
for ax in axes:
    ax.tick_params(axis='both', which='major', labelsize=30)

# 展示图片
plt.tight_layout()

# 保存为eps文件
plt.savefig('/Users/yuanqin/Desktop/Paper/Prompt Tuning EM/submission in revision/figures/exp_sun_alpha.eps', format='eps')


