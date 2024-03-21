from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def prepare_dataset_for_lstm(series, seq_length: int = 4):
    setX: list = []
    setY: list = []
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length]
        future = series[i + 1 : i + seq_length + 1]
        setX.append(past)
        setY.append(future)
    return tensor(setX), tensor(setY)


class DS_LSTM(Module):
    def __init__(self, train, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=len(train) // 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit(self):
        self.train()
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def predict(self, X):
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]

# from torch import no_grad, tensor
# from torch.nn import LSTM, Linear, Module, MSELoss
# from torch.optim import Adam
# from torch.utils.data import DataLoader, TensorDataset


# def prepare_dataset_for_lstm(series, seq_length: int = 4):
#     setX: list = []
#     setY: list = []
#     for i in range(len(series) - seq_length):
#         past = series[i : i + seq_length]
#         future = series[i + 1 : i + seq_length + 1]
#         setX.append(past)
#         setY.append(future)
#     return tensor(setX), tensor(setY)


# class DS_LSTM(Module):
#     def __init__(self, train, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
#         super().__init__()
#         self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self.linear = Linear(hidden_size, 1)
#         self.optimizer = Adam(self.parameters())
#         self.loss_fn = MSELoss()

#         trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
#         self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=len(train) // 10)
#         # self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=10)

#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x[:, -1, :]
#         # return x

#     def fit(self):
#         self.train()
#         for batchX, batchY in self.loader:
#             y_pred = self(batchX)
#             # loss = self.loss_fn(y_pred, batchY)
#             loss = self.loss_fn(y_pred, batchY[:,-1,:])
#             self.optimizer.zero_grad()
#             try:
#                 loss.backward()
#             except:
#                 print('hi')
#             self.optimizer.step()
#         return loss

#     def predict(self, X):
#         with no_grad():
#             y_pred = self(X)
#         # return y_pred[:, -1, :]
#         return y_pred
