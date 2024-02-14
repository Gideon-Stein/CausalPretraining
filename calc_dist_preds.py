import argparse
from model.model_wrapper import Architecture_PL
import torch
import numpy as np
import pickle
from helpers.tools import *
from data.generator import load_deterministic_ds

import numpy as np

# Currently unused and unchecked. This is used to calculate distributions for every link. (appendix)


def get_link_dist(model, batch, window_size=100, samples=50):
    model = model.eval()
    inputs = batch
    if model.corr_input:
        in1 = inputs[0]
        in2 = inputs[1]
    else:
        in1 = inputs
    out = []
    for x in range(samples):
        ind = torch.randint(0, in1.shape[1] - window_size, (1,))
        if model.corr_input:
            y_ = model([in1[:, ind : ind + window_size].detach(), in2.detach()])
        else:
            y_ = model(in1[:, ind : ind + window_size].detach())
        out.append(y_.detach().numpy())
    result = np.stack(out, axis=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    # This is the best transformer of the exp 2 /big grid search
    parser.add_argument(
        "--path",
        default="pretrained_weights/transformer.ckpt",
    )
    parser.add_argument("--dataset", default="XLNL")
    parser.add_argument("--s", default=100, type=int)  # all additional
    parser.add_argument("--bs", default=500, type=int)  # batchsize
    parser.add_argument("--ws", default=100, type=int)  # subsample size of timeseries

    args = parser.parse_args()

    model = Architecture_PL.load_from_checkpoint(args.path)
    M = model.model
    M = M.to("cpu")
    M = M.eval()

    tr, val, te = load_deterministic_ds(
        args.dataset, corr_input=M.corr_input, batch_size=args.bs
    )

    out = {}

    print("tr")
    # one batch is enough for now.
    X, Y = next(iter(tr))
    preds = get_link_dist(
        M,
        X.detach() if not M.corr_input else (X[0].detach(), X[1].detach()),
        samples=args.s,
        window_size=args.ws,
    )
    out["tr"] = [preds, Y]

    print("te")
    X, Y = next(iter(te))
    preds = get_link_dist(
        M,
        X.detach() if not M.corr_input else (X[0].detach(), X[1].detach()),
        samples=args.s,
        window_size=args.ws,
    )
    out["test"] = [preds, Y]

    print("te2")
    tr, val, te = load_deterministic_ds(
        args.dataset + "_additional", corr_input=M.corr_input, batch_size=args.bs
    )
    X, Y = next(iter(te))
    preds = get_link_dist(
        M,
        X.detach() if not M.corr_input else (X[0].detach(), X[1].detach()),
        samples=args.s,
        window_size=args.ws,
    )
    out["test2"] = [preds, Y]

    pickle.dump(out, open("scores/dist_preds_ablation_2.p", "wb"))


if __name__ == "__main__":
    main()
