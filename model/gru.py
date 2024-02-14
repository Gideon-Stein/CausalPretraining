import torch
import torch.nn as nn


class gru(torch.nn.Module):
    def __init__(
        self,
        n_vars=3,
        max_lags=3,
        hidden_size1=10,
        hidden_size2=10,
        hidden_size3=10,
        num_layers=10,
        corr_input=True,
        regression_head=False,
        direction="bidirectional",
    ):
        super(gru, self).__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.regression_head = regression_head
        self.max_lags = max_lags
        self.corr_input = corr_input
        self.n_vars = n_vars
        if direction == "bidirectional":
            self.bidirectional = True
        else:
            self.bidirectional = False

        self.rnn = torch.nn.GRU(
            self.n_vars,
            hidden_size=self.hidden_size1,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        # GRU produces *2 if bidirectional we use two outputs so *4
        if self.bidirectional:
            self.fc1 = torch.nn.Linear(self.hidden_size1 * 4, self.hidden_size2)
        else:
            self.fc1 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        if self.corr_input:
            self.fc2 = torch.nn.Linear(
                self.hidden_size2 + (self.n_vars**2 * self.max_lags), self.hidden_size3
            )
        else:
            self.fc2 = torch.nn.Linear(self.hidden_size2, self.hidden_size3)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_size2)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size3)

        self.fc3 = torch.nn.Linear(self.hidden_size3, self.n_vars**2 * self.max_lags)

        if self.regression_head:
            self.fc4 = torch.nn.Linear(self.hidden_size3, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def reformat(self, x):
        return torch.reshape(x, (x.shape[0], self.n_vars, self.n_vars, self.max_lags))

    def forward(self, x):

        if self.corr_input:
            x, x_corr = x

        output, hidden = self.rnn(x)
        relu = self.relu(output)

        if self.bidirectional:
            relu_select = torch.concat([relu[:, 0, :], relu[:, -1, :]], dim=1)
        else:
            relu_select = relu[:, -1, :]

        hidden1 = self.fc1(relu_select)
        relu1 = self.relu(hidden1)
        bn1 = self.batch_norm1(relu1)

        if self.corr_input:
            hidden2 = self.fc2(
                torch.concat((bn1, x_corr.reshape(bn1.shape[0], -1)), dim=1)
            )
        else:
            hidden2 = self.fc2(bn1)
        relu2 = self.relu(hidden2)
        bn2 = self.batch_norm2(relu2)

        hidden3 = self.fc3(bn2)
        if self.regression_head:
            hidden4 = self.fc4(bn2)
            relu4 = self.relu(hidden4)

        if self.regression_head:
            return (self.reformat(hidden3), relu4)

        else:
            return self.reformat(hidden3)
