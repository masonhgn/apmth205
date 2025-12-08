import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
try:
    df_trace = pd.read_csv('convergence_trace.csv')
    df_heatmap = pd.read_csv('heatmap_data.csv')
except FileNotFoundError:
    print("Error: Ensure CSV files are in the current directory.")
    exit()

# --- Figure 1: Convergence Trace (Log Scale) ---
plt.figure(figsize=(10, 6))

# Clean 0 values for Log Plot (Machine Precision)
df_trace['Standard_Error'] = df_trace['Standard_Error'].replace(0, 1e-18)
df_trace['GCMH_Error'] = df_trace['GCMH_Error'].replace(0, 1e-18)

plt.plot(df_trace['Iteration'], df_trace['Standard_Error'], 
         label='Standard Hybrid (Newton+Bisect)', color='black', linestyle='--', marker='o')
plt.plot(df_trace['Iteration'], df_trace['GCMH_Error'], 
         label='GCM-H (Guarded Halley)', color='red', linewidth=2.5, marker='s')

plt.yscale('log')
plt.xlabel('Iteration Count', fontsize=12)
plt.ylabel('Absolute Error $|C_{model} - C_{mkt}|$', fontsize=12)
plt.title('Convergence Comparison: Deep OTM Call ($K=140, T=0.1$)', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=12)
plt.axhline(y=1e-8, color='green', linestyle=':', label='Tolerance (1e-8)')
plt.tight_layout()
plt.savefig('figure1_convergence.png', dpi=300)
print("Created figure1_convergence.png")

# --- Figure 2: Stability Heatmap ---
# Pivot for heatmap structure
hm_std = df_heatmap.pivot(index='Time', columns='Moneyness', values='Iter_Standard')
hm_new = df_heatmap.pivot(index='Time', columns='Moneyness', values='Iter_GCMH')

# Shared Color Scale for Fair Comparison
vmin = min(df_heatmap['Iter_Standard'].min(), df_heatmap['Iter_GCMH'].min())
vmax = max(df_heatmap['Iter_Standard'].max(), df_heatmap['Iter_GCMH'].max())

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

sns.heatmap(hm_std, ax=axes[0], cmap='magma_r', vmin=vmin, vmax=vmax, cbar=False)
axes[0].set_title('Standard Hybrid Solver', fontsize=14)
axes[0].set_xlabel('Moneyness ($S/K$)', fontsize=12)
axes[0].set_ylabel('Time to Maturity ($T$)', fontsize=12)
axes[0].invert_yaxis() # Cartesian style (T=0.1 at bottom)

sns.heatmap(hm_new, ax=axes[1], cmap='magma_r', vmin=vmin, vmax=vmax, 
            cbar_kws={'label': 'Iterations'})
axes[1].set_title('GCM-H (Proposed)', fontsize=14)
axes[1].set_xlabel('Moneyness ($S/K$)', fontsize=12)
axes[1].set_ylabel('')
axes[1].invert_yaxis()

plt.suptitle('Global Stability: Iteration Count Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('figure2_heatmap.png', dpi=300)
print("Created figure2_heatmap.png")