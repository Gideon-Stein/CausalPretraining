import pickle
from generator import graph_generator
import os
import torch
import argparse


def shuffle_links(d):
    a = torch.randperm(d[1].shape[0])
    b = torch.randperm(d[1].shape[1])
    c = torch.randperm(d[1].shape[2])
    d_random = d[1][a][:, b][:, :, c]
    d = (d[0], d_random)
    return d


def make_deterministic_ds(
    epoch_size=300,
    n_vars=3,
    max_lags=2,
    link_threshold=0.85,
    param_range=[0.3, 0.5],
    test_range=[0.2, 0.3],
    val_range=[0.2, 0.3],
    restrict_links=True,
    noise_var=0.4,
    noise_test=0.6,
    noise_val=0.4,
    time_series_n=500,
    binary=False,
    tr_n=5000,
    te_n=500,
    eval_n=500,
    name="base_train",
    min_max_scaling=True,
    threshold_filter=[-10, 10],
    random=False,
    variable_noise=False,
    functional_class="linear",
    mirror_range=False,
    n_fixed_labels=0,
    distinguish_mode=False,
):
    """Generates synthetic datasets with various specifications and using the generator class.

    Creates folder with train/test/val/args.

    Args:
        epoch_size: Not used. Necessary for on the fly data generation.
        n_vars: N variables involved in the sample
        max_lags: The maximum lag that is possible
        link_threshold: how many links of the fully connected graph are on average 0
        param_range: range from which values in A a drawn from
        test_range: range from which values in A a drawn from
        val_range: range from which values in A a drawn from
        restrict_links: Only allow for a single lag from each variable (increases stability of the processes)
        noise_var: Variance of the noise added to the var samples
        noise_test: Variance of the noise added to the test samples
        noise_val: Variance of the noise added to the train samples
        time_series_n: length of the sample time series:
        binary: Labels are exported as binary
        tr_n: n samples training
        te_n: n samples test
        eval_n: n samples validation
        name: name of the dataset (used for folder)
        min_max_scaling: whether to scale the sample between 0 and 1
        threshold_filter: filter out samples with values higher/lower than threshold
        random: Completely randomize the labels after generating
        variable_noise: Change noise variance randomly every 100 steps
        functional_class: set of functions that is used for drawing the relationship between variables
        mirror_range: also possibly use the negative values of the specified param ranges
        n_fixed_label: Only generate a fixed set of labels for all samples
        distinguish_mode: Generate a dataset with half linear and half nonlinear samples (used in previous experiments.)
    """

    p = "deterministic_ds/" + name
    isExist = os.path.exists(p)
    if not isExist:
        os.makedirs(p)

        pickle.dump(locals(), open(p + "/args.p", "wb"))

        A = graph_generator(
            epoch_size=epoch_size,
            n_vars=n_vars,
            max_lags=max_lags,
            link_threshold=link_threshold,
            param_range=param_range,
            restrict_links=restrict_links,
            noise_var=noise_var,
            time_series_n=time_series_n,
            binary=binary,
            min_max_scaling=min_max_scaling,
            threshold_filter=threshold_filter,
            variable_noise=variable_noise,
            functional_class=functional_class,
            mirror_range=mirror_range,
            return_nonlinear=True,
            n_fixed_labels=n_fixed_labels,
        )
        ds = []
        ds_val = []
        ds_test = []

        # Check whether the specified path exists or not
        if distinguish_mode:
            for x in range(int(tr_n / 2)):
                d = A.__getitem__(10)
                ds.append(d)
            A.param_range = val_range
            A.noise_var = noise_val
            for x in range(int(eval_n / 2)):
                d = A.__getitem__(10)
                ds_val.append(d)
            A.param_range = test_range
            A.noise_var = noise_test
            for x in range(te_n):
                d = A.__getitem__(10)
                ds_test.append(d)
            A.param_range = param_range
            A.noise_var = noise_var
            print("changing functional class")
            A.functional_class = "nonlinear"
            A.nonlinear_threshold = 1
            A.func_options = A.init_func_options()
            for x in range(int(tr_n / 2)):
                d = A.__getitem__(10)
                ds.append(d)
            A.param_range = val_range
            A.noise_var = noise_val
            for x in range(int(eval_n / 2)):
                d = A.__getitem__(10)
                ds_val.append(d)
            A.param_range = test_range
            A.noise_var = noise_test
            for x in range(te_n):
                d = A.__getitem__(10)
                ds_test.append(d)

        else:
            for x in range(tr_n):
                d = A.__getitem__(10)
                if random:
                    # shuffle the links
                    d = shuffle_links(d)
                ds.append(d)

            A.param_range = val_range
            A.noise_var = noise_val
            for x in range(eval_n):
                d = A.__getitem__(10)
                if random:
                    # shuffle the links
                    d = shuffle_links(d)
                ds_val.append(d)

            A.param_range = test_range
            A.noise_var = noise_test
            for x in range(te_n):
                d = A.__getitem__(10)
                if random:
                    # shuffle the links
                    d = shuffle_links(d)
                ds_test.append(d)

        pickle.dump(ds, open(p + "/simple_train.p", "wb"))
        pickle.dump(ds_val, open(p + "/simple_val.p", "wb"))
        pickle.dump(ds_test, open(p + "/simple_test.p", "wb"))
    else:
        print("Dataset exists.")

    # python program to check if a directory exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_up", action="store_true")
    parser.add_argument("--synthetic_six", action="store_true")
    parser.add_argument("--joint", action="store_true")

    args = parser.parse_args()

    if not os.path.exists("deterministic_ds"):
        os.makedirs("deterministic_ds")

    # We always generate two full datasets. From the second one, only test is used during experiments.

    if args.synthetic_six:
        print("Building Scale up dataset")
        # Generate 6 sets with increasing amount of complexity:
        for x in [
            [3, 2, "SL", 0.85, False, "linear"],
            [5, 3, "ML", 0.9, True, "linear"],
            [3, 2, "SNL", 0.85, False, "simple_nonlinear"],
            [5, 3, "MNL", 0.9, True, "simple_nonlinear"],
            [3, 2, "LNL", 0.85, False, "nonlinear"],
            [5, 3, "XLNL", 0.9, True, "nonlinear"],
        ]:
            print("Generating: " + str(x))

            make_deterministic_ds(
                name=x[2],
                max_lags=x[1],
                test_range=[0.2, 0.3],
                val_range=[0.2, 0.3],
                noise_test=0.6,
                n_vars=x[0],
                link_threshold=x[3],
                mirror_range=x[4],
                functional_class=x[5],
            )

            make_deterministic_ds(
                name=x[2] + "_additional",
                max_lags=x[1],
                test_range=[0.3, 0.5],
                val_range=[0.3, 0.5],
                noise_test=0.4,
                n_vars=x[0],
                link_threshold=x[3],
                mirror_range=x[4],
                functional_class=x[5],
            )

    if args.scale_up:
        print("Building Scale up dataset")
        # Generate 5 sets with increasing amount of variables:
        for x in [
            [5, 3, ""],
            [7, 3, "_2"],
            [10, 3, "_3"],
            [12, 3, "_4"],
            [15, 3, "_5"],
        ]:
            print("Generating: " + str(x))
            make_deterministic_ds(
                name="scale_up" + x[2],
                max_lags=x[1],
                n_vars=x[0],
                test_range=[0.2, 0.3],
                val_range=[0.2, 0.3],
                noise_test=0.6,
                mirror_range=True,
                link_threshold=0.9,
            )

            make_deterministic_ds(
                name="scale_up" + x[2] + "_additional",
                max_lags=x[1],
                n_vars=x[0],
                test_range=[0.3, 0.5],
                val_range=[0.3, 0.5],
                noise_test=0.4,
                mirror_range=True,
                link_threshold=0.9,
            )

    if args.joint:

        if not os.path.exists("deterministic_ds/joint"):
            os.makedirs("deterministic_ds/joint")
            os.makedirs("deterministic_ds/joint_additional")

        # all six datasets
        onlyfiles = []
        for x in ["SL", "ML", "SNL", "MNL", "LNL", "XLNL"]:
            onlyfiles.append("deterministic_ds/" + x + "/")
            onlyfiles.append("deterministic_ds/" + x + "_additional/")

        stack = []  # train
        stack2 = []  # val out of range
        stack3 = []  # test out of range
        stack4 = []  # test2 out of range
        for f in onlyfiles:
            print(f)
            if "additional" in f:
                stack.append(pickle.load(open(f + "simple_train.p", "rb")))
                stack.append(pickle.load(open(f + "simple_val.p", "rb")))
                stack4.append(pickle.load(open(f + "simple_test.p", "rb")))
            else:
                stack.append(pickle.load(open(f + "simple_train.p", "rb")))
                stack2.append(pickle.load(open(f + "simple_val.p", "rb")))
                stack3.append(pickle.load(open(f + "simple_test.p", "rb")))

        stack = [item for sublist in stack for item in sublist]
        stack2 = [item for sublist in stack2 for item in sublist]
        stack3 = [item for sublist in stack3 for item in sublist]
        stack4 = [item for sublist in stack4 for item in sublist]

        print(len(stack), len(stack2), len(stack3), len(stack4))
        # Add padding for all samples with less than 5 variables.
        for group in [stack, stack2, stack3, stack4]:
            for x in range(len(group)):
                if group[x][0].shape[1] == 3:
                    a = torch.concat(
                        [group[x][0], torch.normal(0, 0.01, (500, 2))], axis=1
                    )
                    b = torch.nn.functional.pad(
                        group[x][1], (1, 0, 0, 2, 0, 2), mode="constant", value=0.0
                    )
                    group[x] = (a, b)

        print(len(stack), len(stack2), len(stack3), len(stack4))

        # To keep the structure we exort it twice. This should be fixed...
        pickle.dump(stack, open("deterministic_ds/joint/simple_train.p", "wb"))
        pickle.dump(stack2, open("deterministic_ds/joint/simple_val.p", "wb"))
        pickle.dump(stack3, open("deterministic_ds/joint/simple_test.p", "wb"))
        pickle.dump(
            stack, open("deterministic_ds/joint_additional/simple_train.p", "wb")
        )
        pickle.dump(
            stack2, open("deterministic_ds/joint_additional/simple_val.p", "wb")
        )
        pickle.dump(
            stack4, open("deterministic_ds/joint_additional/simple_test.p", "wb")
        )


if __name__ == "__main__":
    main()
