import torch

from configs.NN_config import hparams


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        input_dim = hparams.MODEL.INPUT_DIM
        output_dim = hparams.MODEL.OUTPUT_DIM
        hidden_dim = hparams.MODEL.HIDDEN_DIM
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
