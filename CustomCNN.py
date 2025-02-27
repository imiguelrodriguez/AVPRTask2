# Create a neural net class
import torch
from torch import nn
import torch.nn.functional as F  # Add this line

'''
class CustomCNN(nn.Module):

    # Defining the Constructor
    def __init__(self, num_classes=40):
        super(CustomCNN, self).__init__()

        # In the init function, we define each layer we will use in our model

        # Our images are RGB, so we have input channels = 3.
        # We will apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # A second convolutional layer takes 12 input channels, and generates 24 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # We in the end apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)

        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)

        # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 224/2/2 is 56.
        # This means that our feature tensors are now 56 x 56, and we've generated 24 of them

        # We need to flatten these in order to feed them to a fully-connected layer
        self.fc = nn.Linear(in_features=56 * 56 * 24, out_features=num_classes)

    def forward(self, x):
        # In the forward function, pass the data through the layers we defined in the init function

        # Use a ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))

        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))

        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(-1, 56 * 56 * 24)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        # Return class probabilities via a log_softmax function
        return torch.log_softmax(x, dim=1)
'''
class CustomCNN(nn.Module):
    def __init__(self, num_classes=40, num_layers=2, base_filters=12, kernel_size=3, pool_size=2, dropout_prob=0.2, input_size=224):
        super(CustomCNN, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = 3  # Initial number of input channels (RGB images)
        for i in range(num_layers):
            out_channels = base_filters * (2 ** i)  # Double the filters with each layer
            self.layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1)
            )
            in_channels = out_channels

        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.drop = nn.Dropout2d(p=dropout_prob)

        # Dynamically calculate flattened feature size
        flattened_size = self._get_flattened_size(input_size=input_size)
        self.fc = nn.Linear(flattened_size, num_classes)

    def _get_flattened_size(self, input_size):
        """
        Calculate the flattened size of the tensor after convolution and pooling layers.
        Args:
            input_size (int): The size of the input image (assumes square image).
        Returns:
            int: Flattened size of the tensor.
        """
        x = torch.zeros(1, 3, input_size, input_size)  # Simulate a single input batch
        with torch.no_grad():
            for conv_layer in self.layers:
                x = F.relu(self.pool(conv_layer(x)))
            x = x.view(1, -1)  # Flatten the output
        return x.size(1)

    def forward(self, x):
        for conv_layer in self.layers:
            x = F.relu(self.pool(conv_layer(x)))

        x = F.dropout(self.drop(x), training=self.training)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)
