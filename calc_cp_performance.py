import pickle
import pandas as pd

import torch
import torch.multiprocessing
import argparse

from helpers.tools import *
from helpers.tools import lagged_batch_corr
from model.model_wrapper import Architecture_PL


@timing
def infer(M , data):
    _ = M(data)

def test_time (m ,d, n=100):
    # Running on a single batch now to make it faster.
    model = Architecture_PL.load_from_checkpoint(m, map_location="cpu")
    M = model.model
    M = M.eval()
    X = torch.stack([x[0] for x in d])
    Y = torch.stack([x[1] for x in d])
    corr = lagged_batch_corr(X,3)
    stack = []
    for x in range(n):
        _, time = infer(M,(X, corr))
        stack.append(time)
    return stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rivers", action="store_true")
    parser.add_argument("--aerosols", action="store_true")
    parser.add_argument("--speed", action="store_true")

    parser.add_argument("--path", default="pretrained_weights")

    args = parser.parse_args()

    # Load the best models:
    best = pd.read_csv(
        "experimental_results/synthetic_data_1_joint_grid_results/std/summary_big.csv",
        index_col=0,
    ).T

    stack = []
    for x in best["model.model_type"].unique():
        stack.append(
            best.loc[best["model.model_type"] == x].sort_values(
                "Test1_Auroc", ascending=False
            )[:1]
        )
    print(pd.concat(stack))

    try:
        # Saved best performing checkpoints that resulted in the scores above.
        mlp = args.path + "/mlp.ckpt"
        uni = args.path + "/unidirectional.ckpt"
        bi = args.path + "/bidirectional.ckpt"
        conv = args.path + "/convMixer.ckpt"
        trf = args.path + "/transformer.ckpt"
        best = [mlp, uni, bi, conv, trf]
    except:
        print("Please download the weights.")
        return 0

    auroc = torchmetrics.classification.BinaryAUROC()

    if args.rivers:

        res = []
        data = load_river_data()
        # create proper padding
        a = torch.concat(
            [data[0][0, :, :], torch.normal(0, 0.1, (len(data[0][0]), 2))], axis=1
        )
        a = a.unsqueeze(0)
        corr = lagged_batch_corr(a, 3)

        for x in best:
            model = Architecture_PL.load_from_checkpoint(x)
            M = model.model
            M = M.to("cpu")
            M = M.eval()
            pred = torch.sigmoid(M((a[:, :600, :], corr)))[0, :3, :3, -1]
            res.append(auroc(pred, data[1]))

        print(pd.DataFrame(res, index=["MLP", "uGRU", "bGRU", "CM", "Trf"]))

    if args.aerosols:

        data = pickle.load(open("data/deterministic_ds/aerosol/simple_test.p", "rb"))

        X = data[0][:300]
        Y = data[1][:300]
        X = (X - X.min()) / (X.max() - X.min())
        corr = lagged_batch_corr(X, 3)

        res = []
        for x in best:
            model = Architecture_PL.load_from_checkpoint(x)
            M = model.model
            M = M.to("cpu")
            M = M.eval()
            pred = torch.sigmoid(M((X, corr)))[:, :, :, -1]
            # pred = torch.sigmoid(M((X, corr))).max(axis=3)[0]
            res.append(pred[:, 0, 1] > pred[:, 1, 0])

        print(pd.DataFrame(
            [(x.sum() / len(x)).numpy() for x in res],
            index=["MLP", "uGRU", "bGRU", "CM", "Trf"],
        ).T)

    if args.speed:
        data1 = pickle.load(open("data/deterministic_ds/scale_up/simple_test.p", "rb"))
        # Only the large moodels are included in the repo so the inference speed here is only performed with 5 variables.
        # in any case It should make sense that it scales well :)

        res = test_time(mlp,data1, n=50)
        print(np.array(res).mean())

        res2 = test_time(uni,data1, n=50)
        print(np.array(res2).mean())

        res3 = test_time(bi,data1, n=50)
        print(np.array(res3).mean())

        res4 = test_time(conv,data1, n=50)
        print(np.array(res4).mean())

        res5 = test_time(trf,data1, n=50)
        print(np.array(res5).mean())

        pickle.dump([res,res2,res3,res4], open("scores/cp_inference_speed.p", "wb"))

if __name__ == "__main__":
    main()

# We use the best performing checkpoints from all std runs here. They can be downloaded here:
