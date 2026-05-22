import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("outputs/results.csv", index_col=0)

plt.figure(figsize=(8, 5))

plt.plot(df.index, df["mAP50"], marker="o")

plt.xlabel("Degradation Type")
plt.ylabel("mAP50")
plt.title("YOLO Robustness Under Visual Degradation")

plt.grid(True)

plt.savefig("outputs/map_plot.png")

plt.show()