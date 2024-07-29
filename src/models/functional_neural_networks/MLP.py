import torch
from configs.NN_config import hparams


class MLP(torch.nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) neural network.

    Attributes:
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.
        fc3 (torch.nn.Linear): Third fully connected layer.
    """

    def __init__(self):
        """
        Initializes the MLP with the given input, hidden, and output dimensions
        specified in the hparams configuration.
        """
        super(MLP, self).__init__()
        input_dim = hparams.MODEL.INPUT_DIM
        output_dim = hparams.MODEL.OUTPUT_DIM
        hidden_dim = hparams.MODEL.HIDDEN_DIM

        # Define the layers of the MLP
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        # Apply ReLU activation after the first fully connected layer
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation after the second fully connected layer
        x = torch.relu(self.fc2(x))
        # Pass through the third fully connected layer
        x = self.fc3(x)
        return x
