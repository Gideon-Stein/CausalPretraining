import torch
from torch.utils.data import Dataset
import pickle
import os
import sys

sys.path.append("..")
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from helpers.tools import lagged_batch_corr


class GeneratorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds_name,
        test_ds_name,
        val_ds_name,
        deterministic_ds,
        ds_path=None,
        ds_name=None,
        batch_size: int = 32,
        epoch_size=300,
        n_vars=3,
        max_lags=3,
        link_threshold=0.7,
        param_range=[-0.5, 0.5],
        restrict_links=True,
        noise_var=0.5,
        time_series_n=600,
        binary=False,
        corr_input=False,
        regression=False,
        min_max_scaling=True,
        return_nonlinear=False,
        variable_noise=False,
        mirror_range=False,
        threshold_filter=[-10, 10],
        functional_class="linear",
        distinguish_mode=False,
    ):

        super().__init__()
        self.deterministic_ds = deterministic_ds
        self.time_series_n = time_series_n
        self.epoch_size = epoch_size
        self.n_vars = n_vars
        self.max_lags = max_lags
        self.link_threshold = link_threshold
        self.param_range = param_range
        self.restrict_links = restrict_links
        self.noise_var = noise_var
        self.binary = binary
        self.corr_input = corr_input
        self.batch_size = batch_size
        self.test_ds_name = test_ds_name
        self.val_ds_name = val_ds_name
        self.train_ds_name = train_ds_name
        self.regression = regression
        self.return_nonlinear = return_nonlinear
        self.min_max_scaling = min_max_scaling
        self.threshold_filter = threshold_filter
        self.variable_noise = variable_noise
        self.mirror_range = mirror_range
        self.functional_class = functional_class
        self.distinguish_mode = distinguish_mode

    def setup(self, stage):

        if self.deterministic_ds:
            self.eeg_train = deterministic_ds(
                ds_path=self.train_ds_name,
                binary=self.binary,
                regression=self.regression,
                corr_input=self.corr_input,
                return_nonlinear=self.return_nonlinear,
                distinguish_mode=self.distinguish_mode,
            )

        else:
            self.eeg_train = graph_generator(
                epoch_size=self.epoch_size,
                n_vars=self.n_vars,
                max_lags=self.max_lags,
                link_threshold=self.link_threshold,
                param_range=self.param_range,
                restrict_links=self.restrict_links,
                noise_var=self.noise_var,
                time_series_n=self.time_series_n,
                binary=self.binary,
                min_max_scaling=self.min_max_scaling,
                threshold_filter=self.threshold_filter,
                variable_noise=self.variable_noise,
                functional_class=self.functional_class,
                mirror_range=self.mirror_range,
                return_nonlinear=self.return_nonlinear,
                # no distinguish mode!
            )

        self.eeg_val = deterministic_ds(
            ds_path=self.val_ds_name,
            binary=self.binary,
            regression=self.regression,
            corr_input=self.corr_input,
            return_nonlinear=self.return_nonlinear,
            distinguish_mode=self.distinguish_mode,
        )
        self.eeg_test = deterministic_ds(
            ds_path=self.test_ds_name,
            binary=self.binary,
            regression=self.regression,
            corr_input=self.corr_input,
            return_nonlinear=self.return_nonlinear,
            distinguish_mode=self.distinguish_mode,
        )
        self.eeg_predict = deterministic_ds(
            ds_path=self.test_ds_name,
            binary=self.binary,
            regression=self.regression,
            corr_input=self.corr_input,
            return_nonlinear=self.return_nonlinear,
            distinguish_mode=self.distinguish_mode,
        )

    def train_dataloader(self):
        return DataLoader(
            self.eeg_train, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.eeg_val, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.eeg_test, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.eeg_predict, batch_size=self.batch_size, num_workers=8, shuffle=True
        )

    def teardown(self, stage):
        pass


class graph_generator(Dataset):
    def __init__(
        self,
        epoch_size=300,
        n_vars=3,
        max_lags=4,
        link_threshold=0.7,
        param_range=[0.3, 0.5],
        mirror_range=False,
        restrict_links=True,
        noise_var=0.5,
        time_series_n=600,
        binary=False,
        min_max_scaling=True,
        threshold_filter=[-10, 10],
        variable_noise=False,
        functional_class="linear",
        nonlinear_threshold=0.5,
        return_nonlinear=False,
        n_fixed_labels=0,
    ):
        self.time_series_n = time_series_n
        self.length = epoch_size
        self.n_vars = n_vars
        self.max_lags = max_lags
        self.link_threshold = link_threshold
        self.param_range = param_range
        self.restrict_links = restrict_links
        self.noise_var = noise_var
        self.binary = binary
        self.scale = min_max_scaling
        self.threshold_filter = threshold_filter
        self.variable_noise = variable_noise
        self.functional_class = functional_class
        self.nonlinear_threshold = nonlinear_threshold
        self.return_nonlinear = return_nonlinear
        self.n_fixed_labels = n_fixed_labels
        self.func_options = self.init_func_options()
        self.saved_labels = []
        self.mirror_range = mirror_range

    def init_func_options(self):
        if self.functional_class == "linear":
            return None
        elif self.functional_class == "simple_nonlinear":
            options = []
            options.append(lambda a: torch.pow(a, 2))
            options.append(torch.exp)
            return options

        elif self.functional_class == "nonlinear":
            options = []
            options.append(lambda a: torch.pow(a, 2))
            options.append(torch.exp)
            options.append(torch.sigmoid)
            options.append(torch.sin)
            options.append(torch.cos)
            options.append(torch.nn.functional.relu)
            options.append(torch.nn.functional.logsigmoid)
            options.append(torch.abs)
            options.append(lambda a: torch.clamp(a, -0.5, 0.5))
            options.append(lambda a: torch.divide(1, a))
            return options

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.random_links()

        func = self.get_random_functional(label)
        ts = self.links_to_ts(label, func)

        # rescale and filter divergent processes.
        if (
            (ts.max() > self.threshold_filter[1])
            or (ts.min() < self.threshold_filter[0])
            or torch.any(torch.isnan(ts))
        ):
            # print("resample nonstationary ts.")
            if self.return_nonlinear:
                ts, label, func = self.__getitem__(12)
            else:
                ts, label = self.__getitem__(12)

        # Test if the problem is ill-conditioned.
        if torch.linalg.matrix_rank(torch.cov(ts.T)) < self.n_vars:
            print("YEET")

        if self.scale:
            maxi = torch.max(ts, axis=0)[0]
            mini = torch.min(ts, axis=0)[0]
            ts = (ts - mini) / (maxi - mini)
            # somehow this sometimes dies.
            if ts.isnan().sum() > 0:
                print("resample because nan")
                if self.return_nonlinear:
                    ts, label, func = self.__getitem__(12)
                else:
                    ts, label = self.__getitem__(12)

        if self.binary:
            label = (label > 0).float()

        if self.return_nonlinear:
            return ts, label, func
        else:
            return ts, label

    def get_random_functional(self, links):

        # current functional options:

        if self.functional_class == "linear":
            return None
        else:
            draw = int(len(self.func_options) / self.nonlinear_threshold)
            return torch.randint(0, draw, links.shape)

    def random_links(
        self,
    ):
        params = torch.zeros((self.n_vars, self.n_vars, self.max_lags))
        # takes in a 3dim matrix and adds random elements
        # check which links should be added:
        # links represent caused, cause, lag( beginning with max lag)
        if self.n_fixed_labels != 0:
            if len(self.saved_labels) < self.n_fixed_labels:
                print("save label")
                self.saved_labels.append(
                    torch.rand(size=params.shape) > self.link_threshold
                )
            condition = self.saved_labels[torch.randint(len(self.saved_labels), (1,))]
        else:
            condition = torch.rand(size=params.shape) > self.link_threshold

        add = (self.param_range[1] - self.param_range[0]) * torch.rand(
            size=params.shape
        ) + self.param_range[0]
        if self.mirror_range:
            sign_flip = (torch.rand(add.shape) > 0.5).type(torch.DoubleTensor)
            sign_flip[sign_flip == 0] = -1
            add = (add * sign_flip).type(torch.float32)

        # restrict the process to only one lag for each cause
        # There is for sure a better way to do this. Quick fix
        if self.restrict_links:
            for x in range(condition.shape[0]):
                for y in range(condition.shape[1]):
                    if sum(condition[x, y, :]) > 1:
                        replace = torch.zeros(condition.shape[2], dtype=torch.bool)
                        replace[
                            torch.randint(low=0, high=condition.shape[2], size=(1,))
                        ] = True
                        condition[x, y, :] = replace
        # restrict condition to one lag from the same variable
        return params + add * condition

    def apply_function_via_mask(self, T, func, apply_mask):
        # Applies a function to
        # elemenets specified my mask and returns the new tensor
        new = func(T)
        return torch.where(apply_mask, new, T)

    def apply_random_nonlinear(self, inT, function_selection):
        # selects a random option with given p > threshold and applies it to the original tensor
        T = torch.clone(inT)
        for x in range(len(self.func_options)):
            selection = function_selection == x
            T = self.apply_function_via_mask(T, self.func_options[x], selection)
        return T

    def random_poly(
        self,
        links,
    ):
        # no link has to be exponent 1 to keep the 0 (0^0 = 1)
        # samples a random polinomial term in range for the links that exists
        exists = links > 0
        condition = torch.rand(size=links.shape) < self.polynominal_chance

        add = (self.polynominal_range[1] - self.polynominal_range[0]) * torch.rand(
            size=links.shape
        ) + self.polynominal_range[0]
        return torch.where(exists * condition, add, 1)

    def links_to_ts(self, links, func=None):

        current_noise = self.noise_var
        # init the time series:
        lag = links.shape[2]
        out = torch.zeros((self.time_series_n + lag, len(links)))
        # Init max_lag with random noise
        init = torch.zeros((lag, len(links)), dtype=torch.float64) + torch.randn(
            (lag, len(links))
        )
        out[:lag] = init
        # Calculate the next time step every time
        # t-lag from ts dot product with the cause and lag dimension
        for x in range(self.time_series_n):

            if self.functional_class != "linear":
                intermediate = self.apply_random_nonlinear(
                    out[x : x + lag].T.unsqueeze(0).repeat(self.n_vars, 1, 1), func
                )

            else:
                intermediate = out[x : x + lag].T.unsqueeze(0).repeat(self.n_vars, 1, 1)
            step = (links * intermediate).sum(1).sum(1)
            if self.variable_noise:
                if (x % 100) == 0:
                    current_noise = self.noise_var * (torch.rand(1) * 1.5)

            step += torch.zeros(len(links), dtype=torch.float64) + (
                current_noise**0.5
            ) * torch.randn(len(links))
            out[x + lag] = step
        # remove the first time points to keep only points that dont use init state
        return out[lag:]


# TODO. I need to generate this for all possible dimensions.
class deterministic_ds(Dataset):
    def __init__(
        self,
        ds_path="single",
        binary=True,
        regression=False,
        corr_input=True,
        return_nonlinear=False,
        distinguish_mode=False,
    ):
        self.ds_path = ds_path
        self.ds = self.load()
        self.binary = binary
        self.regression = regression
        self.corr_input = corr_input
        self.return_nonlinear = return_nonlinear
        self.distinguish_mode = distinguish_mode

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        if self.distinguish_mode:
            X = self.ds[idx][0]
            if self.corr_input:
                corr = lagged_batch_corr(X.unsqueeze(0), self.ds[idx][1].shape[-1])
                X = (X, corr)
            if torch.is_tensor(self.ds[idx][2]):
                # one for nonlinear 0 for linear
                Y = torch.Tensor([1])
            else:
                Y = torch.Tensor([0])
            return X, Y

        else:
            X = self.ds[idx][0]
            Y = self.ds[idx][1]
            if self.return_nonlinear:
                Z = self.ds[idx][2]

            if self.corr_input:
                corr = lagged_batch_corr(X.unsqueeze(0), Y.shape[-1])
                X = (X, corr)

            if self.binary:
                Y_clas = (torch.abs(Y) > 0).float()
            else:
                Y_clas = Y.float()

            if self.regression:
                Y_reg = torch.unsqueeze(torch.sum(Y_clas > 0), dim=-1).float()
                # print("critical high value!")
                if self.return_nonlinear:
                    return X, (Y_clas, Y_reg), Z
                else:
                    return X, (Y_clas, Y_reg)
            else:
                if self.return_nonlinear:
                    return X, Y_clas, Z
                else:
                    return X, Y_clas

    def load(self):
        path = os.path.join(self.ds_path)
        if os.path.isfile(path):
            if "kuramoto" in path:
                return torch.load(path)
            else:
                return pickle.load(open(path, "rb"))
        else: 
            raise FileNotFoundError(path)


def load_deterministic_ds(
    ds_name,
    binary=True,
    return_nonlinear=False,
    corr_input=True,
    regression=False,
    distinguish_mode=False,
    batch_size=500,
    only_test=False,
):
    path = "data/deterministic_ds/" + ds_name + "/"
    if only_test:
        mod = GeneratorDataModule(
            train_ds_name=path + "simple_test" + ".p",
            test_ds_name=path + "simple_test" + ".p",
            val_ds_name=path + "simple_test" + ".p",
            deterministic_ds=True,
            binary=binary,
            batch_size=batch_size,
            corr_input=corr_input,
            regression=regression,
            return_nonlinear=return_nonlinear,
            distinguish_mode=distinguish_mode,
        )
    else:
        mod = GeneratorDataModule(
            train_ds_name=path + "simple_train" + ".p",
            test_ds_name=path + "simple_test" + ".p",
            val_ds_name=path + "simple_val" + ".p",
            deterministic_ds=True,
            binary=binary,
            batch_size=batch_size,
            corr_input=corr_input,
            regression=regression,
            return_nonlinear=return_nonlinear,
            distinguish_mode=distinguish_mode,
        )
    mod.setup(None)
    return mod.train_dataloader(), mod.val_dataloader(), mod.test_dataloader()
