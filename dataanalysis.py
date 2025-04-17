import os
import os.path as osp
import json

from config import PROMPT_LEN_LOG_DIR, FIG_LOG_DIR

import matplotlib.pyplot as plt
import numpy as np


def save_hist_with_fit(
    lengths_list,
    output_dir=FIG_LOG_DIR,
    titles=None,
    bins='auto'
):
    os.makedirs(output_dir, exist_ok=True)

    for i, lengths in enumerate(lengths_list):
        lengths = np.array(lengths)
        counts, bin_edges = np.histogram(lengths, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(bin_centers, counts, width=np.diff(bin_edges), align='center',
               color='skyblue', edgecolor='black', alpha=0.6, label="Histogram")
        ax.plot(bin_centers, counts, 'o-', color='red', label="Empirical Fit")

        title = titles[i] if titles else f"Plot {i+1}"
        ax.set_title(title)
        ax.set_xlabel("String Length")
        ax.set_ylabel("Probability Density")
        ax.grid(True)
        ax.legend()

        filename = os.path.join(output_dir, f"plot_{title}.png")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)


with open(osp.join(PROMPT_LEN_LOG_DIR, "prompt_set2.json"), "r") as f:
    resd = json.load(f)

data_l = []
titile_l = []
for k, v in resd.items():
    print(k)
    print(len(v))
    print(max(v))
    print(min(v))
    data_l.append(v)
    titile_l.append(f"String Length Distribution for {k}")

save_hist_with_fit(data_l, titles=titile_l)
