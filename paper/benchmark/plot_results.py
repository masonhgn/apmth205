import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility
except ImportError:
    print("Please install py_vollib: pip install py_vollib")
    exit()


sns.set_theme(style="whitegrid")

# validation
print("Validating against Jäckel (py_vollib)...")
df_acc = pd.read_csv("data_accuracy.csv")
sota_vols = []

for _, row in df_acc.iterrows():
    #jackel to solve same problem
    try:
        #call option
        v = implied_volatility(row['Price'], row['S'], row['K'], row['T'], row['r'], 'c')
    except:
        v = np.nan
    sota_vols.append(v)

df_acc['SOTA_Vol'] = sota_vols
df_acc['Error'] = (df_acc['Solved_GCMH'] - df_acc['SOTA_Vol']).abs()


plt.figure(figsize=(8, 5))
plt.plot(range(len(df_acc)), df_acc['Error'], marker='o', linestyle='--')
plt.yscale('log')
plt.title("Forward Accuracy vs Jäckel (SOTA)", fontsize=14)
plt.ylabel("Absolute Discrepancy", fontsize=12)
plt.xlabel("Test Case Index (Increasing Volatility)", fontsize=12)
plt.axhline(1e-15, color='r', linestyle=':', label='Machine Epsilon')
plt.legend()
plt.tight_layout()
plt.savefig("figure4_accuracy.png", dpi=300)
print(f"Max Discrepancy vs Jäckel: {df_acc['Error'].max()}")


df_trace = pd.read_csv("data_trace.csv")
#clean 0s
df_trace = df_trace.replace(0, 1e-18) 

plt.figure(figsize=(8, 5))
plt.plot(df_trace['Iteration'], df_trace['Err_Standard'], label='Standard Hybrid', marker='o', color='black')
plt.plot(df_trace['Iteration'], df_trace['Err_GCMH'], label='GCM-H', marker='s', color='red')
plt.yscale('log')
plt.title("Convergence Comparison (Deep OTM)", fontsize=14)
plt.ylabel("Pricing Error (Log Scale)", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("figure1_convergence.png", dpi=300)

#heatmap
df_hm = pd.read_csv("data_heatmap.csv")
piv_std = df_hm.pivot(index="Time", columns="Moneyness", values="Iter_Standard")
piv_new = df_hm.pivot(index="Time", columns="Moneyness", values="Iter_GCMH")
vmin, vmax = min(df_hm.min().iloc[2:]), max(df_hm.max().iloc[2:])

fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.heatmap(piv_std, ax=ax[0], vmin=vmin, vmax=vmax, cmap="magma_r", cbar=False)
ax[0].set_title("Standard Hybrid Iterations")
ax[0].invert_yaxis()

sns.heatmap(piv_new, ax=ax[1], vmin=vmin, vmax=vmax, cmap="magma_r", cbar_kws={'label': 'Iterations'})
ax[1].set_title("GCM-H Iterations")
ax[1].invert_yaxis()
plt.tight_layout()
plt.savefig("figure2_heatmap.png", dpi=300)

#efficiency
df_eff = pd.read_csv("data_efficiency.csv")
plt.figure(figsize=(6, 5))
plt.bar(df_eff['Method'], df_eff['SolvesPerSec'], color=['gray', 'red'])
plt.title("Computational Throughput", fontsize=14)
plt.ylabel("Solves per Second", fontsize=12)


imp = (df_eff.iloc[1,2] - df_eff.iloc[0,2]) / df_eff.iloc[0,2] * 100
plt.text(1, df_eff.iloc[1,2], f"+{imp:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("figure3_efficiency.png", dpi=300)

print("All figures generated.")