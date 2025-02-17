#!/usr/bin/env python3

"""
Criterion's default plots are bad for comparing different algorithms, so
use matplotlib instead.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import json


def main():
    output_dir = "plots"
    input_dir = "target/criterion"
    group = "pseudorand setup and query"
    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for bench_name in os.listdir(os.path.join(input_dir, group)):
        if bench_name.isnumeric() or bench_name == "report":
            # skip 0/1/... entries
            continue
        bench_folder = os.path.join(input_dir, group, bench_name)

        bench_results = {}

        for sz in os.listdir(bench_folder):
            if sz == "report":
                continue
            sz = int(sz)

            estimates = open(
                os.path.join(bench_folder, str(sz), "new/estimates.json"), "r"
            ).read()
            estimates = json.loads(estimates)

            assert (
                estimates["median"]["confidence_interval"]["confidence_level"] - 0.95
            ) < 1e-3
            bench_results[sz] = (
                estimates["median"]["point_estimate"],
                estimates["median"]["confidence_interval"]["lower_bound"],
                estimates["median"]["confidence_interval"]["upper_bound"],
            )

        name_parts = tuple(bench_name.split("_"))

        all_results[name_parts] = bench_results

    for category in ["rand32"]:
        fig, ax = plt.subplots()
        line_style = {"query": "--", "setup": "-"}
        color_map = {}

        for (cat, dict_name, step), vals in sorted(all_results.items()):
            if cat != category:
                continue

            if dict_name not in color_map:
                color_map[dict_name] = "C" + str(len(color_map))

            x = sorted(vals.items())
            sz = np.array([a[0] for a in x])
            ests = np.array([a[1] for a in x])

            # Normalize by number of elements processed
            ests /= 2 ** sz[:, np.newaxis]

            ax.errorbar(
                sz,
                ests[:, 0],
                yerr=(ests[:, 0] - ests[:, 1], ests[:, 2] - ests[:, 0]),
                label=(dict_name + " " + step),
                marker=".",
                ls=line_style[step],
                color=color_map[dict_name],
            )

        ax.set_yscale("log")
        ax.set_ylabel("Time per element (ns)")
        xticks = list(range(0, 33, 4))
        ax.set_xticks(xticks)
        ax.set_xticklabels(["$2^{" + str(x) + "}$" for x in xticks])
        ax.grid(visible="minor", lw=0.5)

        ax.set_xlabel("Number of elements")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title("Results for input/query pattern: " + category)
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, category + ".pdf"))


if __name__ == "__main__":
    main()
