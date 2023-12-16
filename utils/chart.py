"""
Reference: 
https://matplotlib.org/stable/plot_types/stats/errorbar_plot.html#sphx-glr-plot-types-stats-errorbar-plot-py
"""

import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('_mpl-gallery')

data = {
    "1": (26.87, 3.39),
    "2A": (17.86, 2.93),
    "2B": (19.23, 3.02),
    "3": (18.50, 2.97),
    "4A": (14.98, 2.73),
    "4B": (15.58, 2.77),
    "5": (17.57, 2.91),
    "6": (15.43, 2.76),
    "7":(16.35, 2.83)
    }

names = list(data.keys())
values = [pair[0] for pair in data.values()]
errs = [pair[1] for pair in data.values()]

fig, ax = plt.subplots()
ax.errorbar(names, values, errs, fmt='o', linewidth=2, capsize=6)
plt.xlabel('Experiment')
plt.ylabel('pass@5')
plt.title('pass@5 Statistical Significance\n(with Wilson Confidence Intervals, 95%)')

# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()