#!/usr/bin/env python3
"""Quick test to verify regplot without confidence interval"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
x = np.random.uniform(0, 1, 20)
y = -0.5 * x + 0.8 + np.random.normal(0, 0.1, 20)

# Create dataframe
df = pd.DataFrame({'x': x, 'y': y})

# Create two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot WITH confidence interval (default)
ax1.scatter(x, y, alpha=0.6)
sns.regplot(data=df, x='x', y='y', scatter=False, color='gray', 
            line_kws={'linestyle': '--', 'alpha': 0.8, 'linewidth': 2},
            ax=ax1)
ax1.set_title('WITH Confidence Interval (Grey Area)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

# Plot WITHOUT confidence interval (ci=None)
ax2.scatter(x, y, alpha=0.6)
sns.regplot(data=df, x='x', y='y', scatter=False, color='gray',
            line_kws={'linestyle': '--', 'alpha': 0.8, 'linewidth': 2},
            ci=None, ax=ax2)
ax2.set_title('WITHOUT Confidence Interval (ci=None)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.savefig('/workspace/GIT_SHENANIGANS/self-obfuscation/test_regplot_comparison.png')
print("Test plot saved to test_regplot_comparison.png")
print("The left plot shows the grey confidence interval area, the right plot has it removed with ci=None")