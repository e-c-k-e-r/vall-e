#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import cfg

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

			for model in args.models:
				if f'{model.name}.{args.xs}' not in row:
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

	if args.min_x is not None:
		for model in args.models:
			df = df[args.min_x < df[f'{model.name}.{args.xs}']]

	if args.max_x is not None:
		for model in args.models:
			df = df[df[f'{model.name}.{args.xs}'] < args.max_x]

	for gtag, gdf in sorted(
		df.groupby("group"),
		key=lambda p: (p[0].split("/")[-1], p[0]),
	):
		for model in args.models:
			x = f'{model.name}.{args.xs}'
			for ys in args.ys:
				y = f'{model.name}.{ys}'

				if gdf[y].isna().all():
					continue

				if args.min_y is not None:
					gdf = gdf[args.min_y < gdf[y]]
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--xs", default="engine_step")
	parser.add_argument("--ys", nargs="+", default="")
	parser.add_argument("--model", nargs="+", default="*")

	parser.add_argument("--min-x", type=float, default=-float("inf"))
	parser.add_argument("--min-y", type=float, default=-float("inf"))
	parser.add_argument("--max-x", type=float, default=float("inf"))
	parser.add_argument("--max-y", type=float, default=float("inf"))

	parser.add_argument("--filename", default="log.txt")
	parser.add_argument("--group-level", default=1)
	args, unknown = parser.parse_known_args()

	path = cfg.rel_path / "logs"
	paths = path.rglob(f"./*/{args.filename}")

	args.models = [ model for model in cfg.model.get() if model.training and (args.model == "*" or model.name in args.model) ]

	if args.ys == "":
		args.ys = ["loss"]

	plot(paths, args)

	out_path = cfg.rel_path / "metrics.png"
	plt.savefig(out_path, bbox_inches="tight")