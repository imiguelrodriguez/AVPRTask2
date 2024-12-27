import os
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # Add this line
from torchvision import transforms


class Stanford40Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (str): Directory where 'train' and 'test' subdirectories are stored.
            split (str): 'train' or 'test' to specify which split to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = self._load_images()

        # Create a mapping from class names to integer labels
        self.class_to_idx = {class_name: idx for idx, class_name in
                             enumerate(sorted(os.listdir(os.path.join(data_dir, split))))}

    def _load_images(self):
        images = []
        split_dir = os.path.join(self.data_dir, self.split)  # Data directory for the specified split (train/test)

        # Iterate over the directories (each directory corresponds to an action)
        for action in os.listdir(split_dir):
            action_dir = os.path.join(split_dir, action)
            if os.path.isdir(action_dir):  # Check if it's a directory (action)
                # Get all the image filenames for the action
                for img_name in os.listdir(action_dir):
                    img_path = os.path.join(action_dir, img_name)
                    if os.path.isfile(img_path):
                        images.append((img_path, action))

        print(f"Loaded {len(images)} images from {self.split} split.")
        return images

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image file path and label from the images list
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB

        if self.transform:
            image = self.transform(image)

        # Convert the label from string to integer using class_to_idx
        label_idx = self.class_to_idx[label]

        return image, label_idx


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the output size after convolution and pooling
        self._dummy_input = torch.zeros(1, 3, 128, 128)  # Dummy input to compute output size
        self._calculate_conv_output()

        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Adjust input size for the fully connected layer
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_conv_output(self):
        # Pass a dummy input through the conv layers to determine the size of the flattened feature map
        x = self._dummy_input
        x = self.pool(F.relu(self.conv1(x)))  # First conv + pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv + pool
        self.fc_input_size = x.numel()  # Calculate the number of features for the fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First convolutional block
        x = self.pool(F.relu(self.conv2(x)))  # Second convolutional block
        x = x.view(-1, self.fc_input_size)  # Flatten the output
        x = F.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Define transforms (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128 for consistency
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Paths to the image directory (data/train and data/test)
data_dir = 'data'  # Make sure the directory is correct for your dataset


# Main execution guard
if __name__ == '__main__':  # Add this to avoid the error on Windows
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the dataset and data loaders
    train_dataset = Stanford40Dataset(data_dir=data_dir, split='train', transform=transform)
    test_dataset = Stanford40Dataset(data_dir=data_dir, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Instantiate the model, loss function, and optimizer
    model = SimpleCNN(num_classes=40)  # Stanford 40 has 40 action classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Beginning training...")
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch}')
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 61:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # Testing the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {(100 * correct / total):.2f}%')

