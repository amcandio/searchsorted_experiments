import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_csv(
    csv_path,
    output_path,
    yscale="log",
    exclude_methods=None,
    include_methods=None,
    max_n=None,
):
    if exclude_methods is None:
        exclude_methods = set()

    df = pd.read_csv(csv_path)

    # Ensure numeric
    df["n"] = pd.to_numeric(df["n"])
    df["time"] = pd.to_numeric(df["time"])

    # Filter methods
    if exclude_methods:
        df = df[~df["method"].isin(exclude_methods)]

    if include_methods:
        df = df[df["method"].isin(include_methods)]

    # Filter maximum input size
    if max_n is not None:
        df = df[df["n"] <= max_n]

    plt.figure(figsize=(8, 6))

    for method, g in df.groupby("method"):
        g = (
            g.groupby("n", as_index=False)["time"]
            .min()
            .sort_values("n")
        )
        plt.plot(
            g["n"],
            g["time"],
            label=method,
        )

    # x-axis: log base 2
    plt.xscale("log", base=2)

    # M1 Pro cache boundaries for np.int32
    # L1 data cache: 128 KB = 2^15 int32 values
    # L2 cache: 12 MB = 1.5 * 2^21 int32 values
    l1_size = 2**15
    l2_size = 1.5 * 2**21

    x_max = df["n"].max()

    if l1_size <= x_max:
        plt.axvline(
            l1_size,
            linestyle="--",
            label=r"L1 D-cache ($2^{15}$ int32)",
        )

    if l2_size <= x_max:
        plt.axvline(
            l2_size,
            linestyle="--",
            label=r"L2 cache ($1.5 \times 2^{21}$ int32)",
        )

    # y-axis
    if yscale == "log":
        plt.yscale("log")
    elif yscale == "linear":
        plt.yscale("linear")
    else:
        raise ValueError(f"Unknown yscale: {yscale}")

    # Power-of-2 ticks
    sizes = df["n"].to_numpy(dtype=float)
    min_exp = int(np.floor(np.log2(sizes.min())))
    max_exp = int(np.ceil(np.log2(sizes.max())))

    xticks = [2**i for i in range(min_exp, max_exp + 1)]

    # Don't show ticks beyond max_n
    if max_n is not None:
        xticks = [x for x in xticks if x <= max_n]

    plt.xticks(
        xticks,
        [f"$2^{{{int(np.log2(x))}}}$" for x in xticks],
    )

    plt.xlabel("Input size (n)")
    plt.ylabel("Time per query (ns)")
    plt.title("Binary search latency per query")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot binary search latency from CSV benchmarks"
    )

    parser.add_argument(
        "--csv",
        default="results.csv",
        help="Path to CSV file (default: results.csv)",
    )

    parser.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="linear",
        help="Y-axis scale (default: linear)",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Methods to exclude from plot",
    )

    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Methods to include from plot",
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Maximum input size to plot",
    )

    parser.add_argument(
        "--output",
        default="benchmark.png",
        help="Output image path (default: benchmark.png)",
    )

    args = parser.parse_args()

    plot_csv(
        csv_path=args.csv,
        output_path=args.output,
        yscale=args.yscale,
        exclude_methods=set(args.exclude),
        include_methods=set(args.include),
        max_n=args.max_n,
    )


if __name__ == "__main__":
    main()