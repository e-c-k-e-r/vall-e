#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(paths, args):
    dfs = []

    for path in paths:
        with open(path, "r") as f:
            text = f.read()

        rows = []

        pattern = r"(\{.+?\})\.\n"

        for row in re.findall(pattern, text, re.DOTALL):
            try:
                row = json.loads(row)
            except Exception as e:
                continue

            for x in args.xs:
                if x not in row:
                    continue
                rows.append(row)
                break

        df = pd.DataFrame(rows)

        if "name" in df:
            df["name"] = df["name"].fillna("train")
        else:
            df["name"] = "train"

        df["group"] = str(path.parents[args.group_level])
        df["group"] = df["group"] + "/" + df["name"]

        dfs.append(df)

    df = pd.concat(dfs)

    if args.max_y is not None:
        for x in args.xs:
            df = df[df[x] < args.max_x]

    for gtag, gdf in sorted(
        df.groupby("group"),
        key=lambda p: (p[0].split("/")[-1], p[0]),
    ):
        for x in args.xs:
            for y in args.ys:
                gdf = gdf.sort_values(x)

                if gdf[y].isna().all():
                    continue

                if args.max_y is not None:
                    gdf = gdf[gdf[y] < args.max_y]

                gdf[y] = gdf[y].ewm(10).mean()

                gdf.plot(
                    x=x,
                    y=y,
                    label=f"{y}",
                    ax=plt.gca(),
                    marker="x" if len(gdf) < 100 else None,
                    alpha=0.7,
                )

    plt.gca().legend(
        loc="center left",
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(1.04, 0.5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xs", nargs="+", default="ar.engine_step")
    parser.add_argument("--ys", nargs="+", default="ar.loss")
    parser.add_argument("--log-dir", default="logs", type=Path)
    parser.add_argument("--out-dir", default="logs", type=Path)
    parser.add_argument("--filename", default="log.txt")
    parser.add_argument("--max-x", type=float, default=float("inf"))
    parser.add_argument("--max-y", type=float, default=float("inf"))
    parser.add_argument("--group-level", default=1)
    parser.add_argument("--model-name", type=str, default="ar")
    parser.add_argument("--filter", default=None)
    args = parser.parse_args()

    paths = args.log_dir.rglob(f"**/{args.filename}")

    if args.filter:
        paths = filter(lambda p: re.match(".*" + args.filter + ".*", str(p)), paths)

    plot(paths, args)

    name = "-".join(args.ys)
    out_path = (args.out_dir / name).with_suffix(".png")
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
