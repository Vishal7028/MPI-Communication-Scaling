import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("timing_data.txt", sep=r"\s+")

df = df.sort_values(["P", "M"])

combinations = df[["P", "M"]].drop_duplicates().values

box_data = []
labels = []

for P, M in combinations:
    times = df[(df["P"] == P) & (df["M"] == M)]["time"].values
    box_data.append(times)
    labels.append(f"P={P}\nM={M}")

fig, ax = plt.subplots(figsize=(12, 6))

ax.boxplot(
    box_data,
    labels=labels,
    showmeans=True
)

ax.set_title("Execution Time vs Number of Processes")
ax.set_xlabel("Process (P) and Data Size (M)")
ax.set_ylabel("Time (seconds)")
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig("boxplot.png", dpi=300)

plt.show()

