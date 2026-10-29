"""Build the shuffled-label control set for the sanity checks.

    python thesis/make_shuffled.py [--source base] [--seed 0]

Takes an original dataset and permutes the labels of the training and validation samples
across samples, so every series is paired with the graph of a different series. The test
set keeps its true labels: a network that learned anything from the inputs, rather than
the label distribution, has nothing to learn here and must stay at chance level on it.

(data/generate_synthetic_ds.py --random instead permutes the axes of each sample's own
graph, which keeps graph statistics tied to the series; this is the stricter control.)
"""
import argparse
import pickle

import torch

from common import PATHS, ds_folder


def shuffle_labels(samples, generator):
    perm = torch.randperm(len(samples), generator=generator)
    return [(s[0], samples[j][1], *s[2:]) for s, j in zip(samples, perm.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="SL", help="thesis name or original folder")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = PATHS["data_root"] / ds_folder(args.source)
    dst = PATHS["derived_root"] / f"{ds_folder(args.source)}_shuffled"
    if dst.exists():
        raise SystemExit(f"{dst} exists, not overwriting")
    dst.mkdir(parents=True)

    g = torch.Generator().manual_seed(args.seed)
    for split in ("train", "val", "test"):
        samples = pickle.load(open(src / f"simple_{split}.p", "rb"))
        if split != "test":
            samples = shuffle_labels(samples, g)
        pickle.dump(samples, open(dst / f"simple_{split}.p", "wb"))
    pickle.dump({"source": str(src), "seed": args.seed, "shuffled": ["train", "val"]}, open(dst / "args.p", "wb"))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
