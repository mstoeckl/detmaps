#!/usr/bin/env python3

"""
Criterion's default plots are bad for comparing different algorithms, so
use matplotlib instead.
"""

import matplotlib
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

    color_map = {}
    line_style = {"chainq": ":", "parq": "--", "setup": "-"}
    markers = {"chainq": ".", "parq": ".", "setup": "p"}
    # the matplotlib tab20 color list, slightly rounded
    color_list = list(matplotlib.colormaps["tab20"].colors[1::2]) + list(
        matplotlib.colormaps["tab20"].colors[0::2]
    )
    for category in ["rand32", "rand64"]:
        fig, ax = plt.subplots()
        fig.set_size_inches(20 / 2.54, 15 / 2.54)

        active_dicts = set()

        for (cat, dict_name, step), vals in sorted(all_results.items()):
            if cat != category:
                continue
            active_dicts.add(dict_name)

            print(category, dict_name, step)

            if dict_name not in color_map:
                color_map[dict_name] = color_list.pop()

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
                marker=markers[step],
                ls=line_style[step],
                color=color_map[dict_name],
            )

        ax.set_yscale("log")
        ax.set_ylabel("Time per element (ns)")
        ax.set_ylim(
            1, 5e4
        )  # set a 'uniform' ylimit to make cross-plot and cross-computer comparison easier
        xticks = list(range(0, 33, 4))
        ax.set_xticks(xticks)
        ax.set_xticklabels(["$2^{" + str(x) + "}$" for x in xticks])
        ax.grid(visible="minor", lw=0.5)

        ax.set_xlabel("Number of elements")
        # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_title("Results for input/query pattern: " + category)

        labels = ["setup", "chainq", "parq"]
        lines = [
            matplotlib.lines.Line2D(
                [0], [0], color="k", ls=line_style[step], marker=markers[step], lw=1.5
            )
            for step in labels
        ]
        for dict_name in sorted(active_dicts):
            lines.append(
                matplotlib.lines.Line2D(
                    [0], [0], color=color_map[dict_name], ls="-", lw=5
                )
            )
            labels.append(dict_name)

        ax.legend(lines, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, category + ".pdf"))


if __name__ == "__main__":
    main()
