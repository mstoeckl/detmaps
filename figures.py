#!/usr/bin/env python3

"""
Criterion's default plots are bad for comparing different algorithms, so
use matplotlib instead.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os
import numpy as np
import json


def main():
    output_dir = "plots"
    input_dir = "target/bench/main"
    all_results = {}
    os.makedirs(output_dir, exist_ok=True)

    for record in os.listdir(os.path.join(input_dir)):
        if not record.endswith(".json"):
            print(record)
            continue
        dict_name, u_bits, sz = record[:-5].split("_")
        sz = int(sz)

        data = json.loads(open(os.path.join(input_dir, record), "r").read())

        assert data["dictionary"] == dict_name
        assert data["u_bits"] == int(u_bits)
        n = data["n_elements"]
        assert data["n_elements"] == sz or data["n_elements"] == 1 << sz

        measurements = data["measurements"]

        name_parts = ("rand" + str(u_bits), dict_name, "maxread")
        if "max_read_operations" in data:
            if name_parts not in all_results:
                all_results[name_parts] = {}
            v = float(data["max_read_operations"])
            all_results[name_parts][n] = (v, v, v)

        name_parts = ("rand" + str(u_bits), dict_name, "memory")
        if "total_memory_usage" in data:
            if name_parts not in all_results:
                all_results[name_parts] = {}
            v = float(data["total_memory_usage"])
            all_results[name_parts][n] = (v, v, v)

        for step in ["setup", "chainq", "parq"]:
            name_parts = ("rand" + str(u_bits), dict_name, step)
            if name_parts not in all_results:
                all_results[name_parts] = {}

            all_results[name_parts][n] = (
                1e9 * (measurements[step]["median"]),
                1e9
                * (measurements[step]["q1"] - measurements[step]["systematic_error"]),
                1e9
                * (measurements[step]["q3"] + measurements[step]["systematic_error"]),
            )
        for step in ["chainq", "parq"]:
            name_parts = ("rand" + str(u_bits), dict_name, step + "x100")
            if name_parts not in all_results:
                all_results[name_parts] = {}

            setup_times = all_results[("rand" + str(u_bits), dict_name, "setup")][n]
            step_times = all_results[("rand" + str(u_bits), dict_name, step)][n]
            factor = 100
            factor_times = (
                setup_times[0] + step_times[0] * factor,
                setup_times[1] + step_times[1] * factor,
                setup_times[2] + step_times[2] * factor,
            )
            all_results[name_parts][n] = factor_times

    line_style = {
        "chainq": ":",
        "parq": "--",
        "setup": "-",
        "chainqx100": "-",
        "parqx100": "-",
        "maxread": "-",
        "memory": "-",
    }
    markers = {
        "chainq": ".",
        "parq": ".",
        "setup": "p",
        "chainqx100": "x",
        "parqx100": "x",
        "maxread": "x",
        "memory": "x",
    }
    subsets = [
        ("all", None),
        ("paper", ["binsearch", "btree", "hash", "oms+disp", "ocm+disp", "oma+disp"]),
    ]

    for subset_name, subset in subsets:
        # the matplotlib tab20 color list, slightly rounded
        color_list = list(matplotlib.colormaps["tab20"].colors[1::2]) + list(
            matplotlib.colormaps["tab20"].colors[0::2]
        )
        color_map = {}
        for category in ["rand32", "rand64"]:
            with matplotlib.backends.backend_pdf.PdfPages(
                os.path.join(output_dir, category + "_" + subset_name + ".pdf")
            ) as pdf:
                for mode in [
                    "all",
                    "setup",
                    "parq",
                    "chainq",
                    "maxread",
                    "memory",
                    "parqx100",
                    "chainqx100",
                ]:
                    print("mode {}, category {}".format(mode, category))

                    fig, ax = plt.subplots()
                    fig.set_size_inches(20 / 2.54, 15 / 2.54)

                    active_dicts = set()

                    line_no = 0
                    for (cat, dict_name, step), vals in sorted(all_results.items()):
                        if cat != category:
                            continue
                        if subset is not None and dict_name not in subset:
                            continue
                        if mode == "all":
                            if step not in ("setup", "parq", "chainq"):
                                continue
                        else:
                            if mode != step:
                                continue

                        active_dicts.add(dict_name)

                        if mode == "all":
                            print(category, dict_name, step)

                        if dict_name not in color_map:
                            color_map[dict_name] = color_list.pop()

                        x = sorted(vals.items())
                        sz = np.array([a[0] for a in x])
                        ests = np.array([a[1] for a in x])

                        # Normalize by number of elements processed
                        if step != "maxread":
                            ests /= sz[:, np.newaxis]
                        else:
                            # offset lines very slightly to distinguish them?
                            # issue: only have O(1) collisions per point in general; this
                            # offsets points inconsistently. The correct thing to do is
                            # perform offset operations globally
                            ests += 0.003 * line_no * (2 * (line_no % 2) - 1)
                            line_no += 1

                        ax.errorbar(
                            np.log2(sz),
                            ests[:, 0],
                            yerr=(ests[:, 0] - ests[:, 1], ests[:, 2] - ests[:, 0]),
                            label=(dict_name + " " + step),
                            marker=markers[step],
                            ls=line_style[step],
                            color=color_map[dict_name],
                        )

                    ax.set_yscale("log")
                    if mode == "maxread":
                        ax.set_ylabel("Number of reads")
                    elif mode == "memory":
                        ax.set_ylabel("Bytes per element")
                    else:
                        ax.set_ylabel("Time per element (ns)")
                    if mode.endswith("x100"):
                        ax.set_ylim(1e2, 5e5)
                    elif mode == "maxread":
                        ax.set_ylim(0.5, 1e2)
                    elif mode == "memory":
                        ax.set_ylim(1e1, 5e4)
                    else:
                        # set a 'uniform' ylimit to make cross-plot and cross-computer comparison easier
                        ax.set_ylim(0.5, 5e4)
                    xticks = list(range(0, 31, 4))
                    ax.set_xlim(-1, 31)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(["$2^{" + str(x) + "}$" for x in xticks])
                    ax.grid(visible="minor", lw=0.5)

                    ax.set_xlabel("Number of elements")

                    if mode == "all":
                        ax.set_title("Results for input/query pattern: " + category)
                    elif mode == "setup":
                        ax.set_title("Setup time results for: " + category)
                    elif mode == "chainq":
                        ax.set_title("Chain query results for: " + category)
                    elif mode == "parq":
                        ax.set_title("Parallel query results for: " + category)
                    elif mode == "chainqx100":
                        ax.set_title(
                            "Workload: setup + 100n chain queries, for " + category
                        )
                    elif mode == "parqx100":
                        ax.set_title(
                            "Workload: setup + 100n parallel queries, for " + category
                        )
                    elif mode == "maxread":
                        ax.set_title(
                            "Worst-case number of unpredictable/uncached reads, for "
                            + category
                        )
                    elif mode == "memory":
                        ax.set_title("Memory usage, for " + category)
                    else:
                        raise NotImplementedError(mode)

                    if mode == "all":
                        labels = ["setup", "chainq", "parq"]
                        lines = [
                            matplotlib.lines.Line2D(
                                [0],
                                [0],
                                color="k",
                                ls=line_style[step],
                                marker=markers[step],
                                lw=1.5,
                            )
                            for step in labels
                        ]
                    else:
                        labels = []
                        lines = []
                    for dict_name in sorted(active_dicts):
                        lines.append(
                            matplotlib.lines.Line2D(
                                [0], [0], color=color_map[dict_name], ls="-", lw=5
                            )
                        )
                        labels.append(dict_name)

                    ax.legend(lines, labels, loc="center left", bbox_to_anchor=(1, 0.5))

                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close()
                    # plt.savefig(os.path.join(output_dir, category + ".pdf"))


if __name__ == "__main__":
    main()
