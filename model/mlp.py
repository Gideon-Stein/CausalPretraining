import torch
import torch.nn as nn


# Naive idea:
# Split ts in batches, flatten them and run them in paralell in the batch dimension to increase efficiency.
class mlp(torch.nn.Module):
    def __init__(
        self,
        n_vars=3,
        max_lags=3,
        max_ts_length=600,
        corr_input=False,
        hidden_size1=150,
        hidden_size2=100,
        hidden_size3=50,
        hidden_size4=50,
        regression_head=False,
        full_representation_mode=False,
    ):
        super(mlp, self).__init__()
        self.max_ts_length = max_ts_length
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.hidden_size4 = hidden_size4
        self.regression_head = regression_head
        self.max_ts_length = max_ts_length
        self.max_lags = max_lags
        self.n_vars = n_vars
        self.corr_input = corr_input
        self.full_representation_mode = full_representation_mode

        self.fc1 = torch.nn.Linear(self.max_ts_length, self.hidden_size1)
        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        if self.full_representation_mode:
            if self.corr_input:
                in_dim = self.hidden_size2 * (self.n_vars * (self.max_lags + 1)) + (
                    self.n_vars**2 * self.max_lags
                )
            else:
                in_dim = self.hidden_size2 * (self.n_vars * (self.max_lags + 1))
        else:
            if self.corr_input:
                in_dim = self.hidden_size2 * self.n_vars + (
                    self.n_vars**2 * self.max_lags
                )
            else:
                in_dim = self.hidden_size2 * self.n_vars
        self.fc3 = torch.nn.Linear(in_dim, self.hidden_size3)
        self.fc4 = torch.nn.Linear(self.hidden_size3, self.hidden_size4)
        self.fc5 = torch.nn.Linear(self.hidden_size4, self.n_vars**2 * self.max_lags)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size2)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size3)
        self.batch_norm4 = nn.BatchNorm1d(self.hidden_size4)

        if self.regression_head:
            self.fc6 = torch.nn.Linear(self.hidden_size4, 1)

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

        # padd with zeros or cut
        if x.shape[1] < self.max_ts_length:
            pad = torch.zeros(
                (x.shape[0], self.max_ts_length - x.shape[1], x.shape[2])
            ).to("cuda" if (x.get_device() != -1) else "cpu")
            x = torch.concat((x, pad), axis=1)
        if x.shape[1] > self.max_ts_length:
            x = x[:, : self.max_ts_length, :]

        stack = []
        # we embed each ts first and then concat them for further proecessing.
        for var in range(x.shape[2]):
            embed1 = self.batch_norm1(self.relu(self.fc1(x[:, :, var])))
            embed2 = self.batch_norm2(self.relu(self.fc2(embed1)))
            stack.append(embed2)
        if self.corr_input:
            stack.append(x_corr.reshape(x_corr.shape[0], -1))
        hidden1 = torch.concat(stack, axis=1)

        hidden2 = self.batch_norm3(self.relu(self.fc3(hidden1)))

        hidden3 = self.batch_norm4(self.relu(self.fc4(hidden2)))

        hidden4 = self.fc5(hidden3)

        if self.regression_head:
            reg_out = self.relu(self.fc6(hidden3))

        if self.regression_head:
            return (self.reformat(hidden4), reg_out)
        else:
            return self.reformat(hidden4)
