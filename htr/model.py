import torch.nn as nn
from torch.nn import Sequential, MaxPool2d, LSTM, Module, Conv2d, ReLU, BatchNorm2d


class LoghiModel9(Module):

    def __init__(self, alphabetSize):
        super(LoghiModel9, self).__init__()

        self.cnn = Sequential(
            Conv2d(in_channels=1, out_channels=24, kernel_size=(3, 3), padding="same"),
            ReLU(),

            BatchNorm2d(24),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding="same"),
            ReLU(),

            BatchNorm2d(48),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(96),

            Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding="same"),
            ReLU(),
            BatchNorm2d(96),

            MaxPool2d(kernel_size=2, stride=2),
        )
        self.rnn = LSTM(768, 256, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(512, alphabetSize)

    def forward(self, batch):
        features = self.cnn(batch)
        batchSize, channels, height, width = features.size()

        features = features.permute(3, 0, 1, 2)
        features = features.reshape(width, batchSize, channels * height)

        encoding, _ = self.rnn(features)
        result = self.linear(encoding)
        return result
