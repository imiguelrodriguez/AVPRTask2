from torch import nn
from torchvision import models


class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Change the first convolutional layer to accept one channel
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Change the input size in the fully connected layer
        self.resnet.fc = nn.Linear(512, 40)  # Assuming you want 10 classes for MNIST

    def forward(self, x):
        return self.resnet(x)
