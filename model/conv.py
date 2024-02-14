import torch
import torch.nn as nn

# Author: Robert Guthrie
import torch
from torch import nn
import lightning.pytorch as pl
from torch import optim, nn, utils, Tensor

# simple decision maker that takes in the embeddings and the Meta data to make a prediction.
# For now, this should predict the outcome. cpc can be estimated since its not relevant at the moment
# This is basically for the trial submission. We shouldnt expect much from this.


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, channels=1, depth=10, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


def ConvMixer1D(dim, channels, depth, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv1d(channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm1d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm1d(dim),
                    )
                ),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(dim),
            )
            for i in range(depth)
        ],
        nn.AdaptiveAvgPool1d((1)),
        nn.Flatten(),
    )


class conv_mixer(torch.nn.Module):
    def __init__(
        self,
        convM_dim=512,
        convM_depth=5,
        kernel_size=12,
        patch_size=12,
        hidden_size1=100,
        n_vars=3,
        corr_input=False,
        max_lags=2,
        regression_head=False,
        conv1D=False,
        full_representation_mode=False,
    ):
        super(conv_mixer, self).__init__()
        self.convM_dim = convM_dim
        self.convM_depth = convM_depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.hidden_size1 = hidden_size1
        self.n_vars = n_vars
        self.corr_input = corr_input
        self.max_lags = max_lags
        self.regression_head = regression_head
        self.conv1D = conv1D
        self.full_representation_mode = full_representation_mode
        if conv1D:
            self.convM = ConvMixer1D(
                channels=(
                    self.n_vars * (self.max_lags + 1)
                    if self.full_representation_mode
                    else self.n_vars
                ),
                dim=self.convM_dim,
                depth=self.convM_depth,
                kernel_size=self.kernel_size,
                patch_size=self.patch_size,
            )

        else:
            self.convM = ConvMixer(
                dim=self.convM_dim,
                depth=self.convM_depth,
                kernel_size=self.kernel_size,
                patch_size=self.patch_size,
            )

        if self.corr_input:
            self.fc1 = torch.nn.Linear(
                self.convM_dim + (self.n_vars**2 * self.max_lags), self.hidden_size1
            )
        else:
            self.fc1 = torch.nn.Linear(self.convM_dim, self.hidden_size1)

        self.fc2 = torch.nn.Linear(self.hidden_size1, self.n_vars**2 * self.max_lags)
        if self.regression_head:
            self.fc3 = torch.nn.Linear(self.hidden_size1, 1)

        self.batch_norm = nn.BatchNorm1d(self.hidden_size1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def reformat(self, x):
        return torch.reshape(x, (x.shape[0], self.n_vars, self.n_vars, self.max_lags))

    def forward(self, x):
        if self.full_representation_mode:
            if self.corr_input:
                x = (
                    torch.concat(
                        [
                            torch.roll(x[0], shifts=y, dims=1)
                            for y in range(self.max_lags + 1)
                        ],
                        axis=2,
                    ),
                    x[1],
                )
            else:
                x = torch.concat(
                    [torch.roll(x, shifts=y, dims=1) for y in range(self.max_lags + 1)],
                    axis=2,
                )

        if self.corr_input:
            x, x_corr = x

        if self.conv1D:
            hidden1 = self.convM(x.swapaxes(1, 2))
        else:
            hidden1 = self.convM(torch.unsqueeze(x.swapaxes(1, 2), 1))
        if self.corr_input:
            hidden2 = self.fc1(
                torch.concat((hidden1, x_corr.reshape(hidden1.shape[0], -1)), dim=1)
            )
        else:
            hidden2 = self.fc1(hidden1)

        bn2 = self.batch_norm(hidden2)
        hidden3 = self.fc2(bn2)

        if self.regression_head:
            hidden4 = self.fc3(bn2)
            relu4 = self.relu(hidden4)

        if self.regression_head:
            return (self.reformat(hidden3), relu4)
        else:
            return self.reformat(hidden3)
