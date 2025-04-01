import pickle
import os
import torch
import argparse
import pandas as pd
import numpy as np


# THE RAW DATA SOURCES SHOULD BE DOWNLOADED AND SPECIFIED (README.MD)
# River data is loaded directly (needs no building.)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuramoto", action="store_true")
    parser.add_argument(
        "--kuramoto_path",
        default="/home/stein/project_resources/AmortizedCausalDiscovery/codebase/data/",
    )

    parser.add_argument("--aerosol", action="store_true")
    parser.add_argument("--aerosol_path", default="four_outputs_liqcf_pacific.csv")

    args = parser.parse_args()

    if not os.path.exists("deterministic_ds"):
        os.makedirs("deterministic_ds")

    if args.kuramoto:
        print("Building Kuramoto dataset")
        p = "deterministic_ds/kuramoto"
        if not os.path.exists(p):
            os.makedirs(p)

        for x in ["train", "valid", "test"]:
            print(x)
            lab = np.load(args.kuramoto_path + "edges_" + x + "_kuramoto5.npy")
            data = np.load(args.kuramoto_path + "feat_" + x + "_kuramoto5.npy")

            lab = torch.Tensor(lab).unsqueeze(3).float()
            data = data[:, :, :, 1]
            data = torch.Tensor(data).swapaxes(1, 2)
            data = list(data)
            lab = list(lab)
            ds = [
                [torch.Tensor(data[x]), torch.Tensor(lab[x])] for x in range(len(lab))
            ]
            if x == "valid":
                torch.save(ds, p + "/simple_" + "val" + ".p")
            else:
                torch.save(ds, p + "/simple_" + x + ".p")

            del ds

    if args.aerosol:
        print("Building Aerosol dataset")
        p = "deterministic_ds/aerosol"
        if not os.path.exists(p):
            os.makedirs(p)

        # aerosol
        data = pd.read_csv(args.aerosol_path, index_col=0)
        print(data)
        # some formatting
        data["key"] = data["lats"].astype(str) + "_" + data["lons"].astype(str)
        data.drop(columns=["lats", "lons"], inplace=True)
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")

        # Check filter options from original paper:
        # This is impossible to remove for time series inference...  Keep it.
        (data["precip"] > 0.05).sum() / len(data)

        # Guess we can filter these. According to the paper
        print(
            (data["tot_aod"] < 0.07).sum() / len(data),
            (data["tot_aod"] > 1.0).sum() / len(data),
        )


        # variable selection
        variables = ["l_re", "tot_aod", "RH850", "EIS", "whoi_sst", "timestamp"]
        locs = data["key"].unique()

        # Check for full timesteps
        example = data.loc[data["key"] == locs[20], variables]
        times = example["timestamp"].sort_values().values
        spread = pd.DataFrame(
            pd.date_range(times[0], times[-1], freq="d"), columns=["timestamp"]
        )
        example = spread.merge(example, on="timestamp", how="left")
        example.reset_index(drop=True, inplace=True)

        example = data.loc[data["key"] == locs[30], variables]

        example_stack = []
        for loc in locs:
            print(loc)
            example = data.loc[data["key"] == loc, variables]
            for x in example["timestamp"].dt.year.unique():
                print(x)
                year = example[example["timestamp"].dt.year == x].copy()
                year["timestamp"] = (
                    pd.to_datetime(year["timestamp"]).dt.round("D").values
                )
                times = pd.to_datetime(year["timestamp"].sort_values().values)
                full_time = pd.date_range(times[0], times[-1], freq="d")
                spread = pd.DataFrame(full_time, columns=["timestamp"])
                year = spread.merge(year, on="timestamp", how="left")
                year.reset_index(drop=True, inplace=True)
                if (year.isnull().sum().max() > 91) or (
                    len(year) < 365
                ):  # 25% missing  max
                    pass
                else:
                    example_stack.append(
                        torch.Tensor(
                            year.interpolate().drop(columns="timestamp").values[:365]
                        )
                    )

        lab = torch.zeros(5, 5, 1)
        lab[0, 1, 0] = 1
        Y = lab
        X = torch.stack(example_stack)
        pickle.dump([X, Y], open(p + "/simple_test.p", "wb"))


if __name__ == "__main__":
    main()
