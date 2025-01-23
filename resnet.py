import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from CustomResNet import CustomResNet
from TrainTestUtils import train, validate, plot_loss, make_predictions, compute_accuracy
from main import Stanford40Dataset

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Paths to the image directory (data/train and data/test)
data_dir = 'data'
LR = 1e-3
BATCH_SIZE = 196
EPOCHS = 50
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

if __name__ == '__main__':  # Add this to avoid the error on Windows
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the dataset and data loaders
    train_dataset = Stanford40Dataset(data_dir=data_dir, split='train', transform=transform)
    test_dataset = Stanford40Dataset(data_dir=data_dir, split='test', transform=transform)

    numTrainSamples = int(len(train_dataset) * TRAIN_SPLIT)
    numValSamples = int(len(train_dataset) * VAL_SPLIT)
    (trainData, valData) = random_split(train_dataset,
                                        [numTrainSamples, numValSamples],
                                        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(valData, batch_size=BATCH_SIZE)

    # Create an instance of the model class and allocate it to the device
    model = CustomResNet().to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_criteria = nn.CrossEntropyLoss()

    # Training Loop
    epoch_nums = []
    training_loss = []
    validation_loss = []

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        # Call the train function
        train_loss = train(model, device, loss_criteria, train_loader, optimizer, epoch)
        # Call the test function
        val_loss = validate(model, device, loss_criteria, val_loader)

        # Track losses and epochs
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(val_loss)

    # Plot training progress
    plot_loss(epoch_nums, training_loss, validation_loss)

    # Testing the Model
    print("Evaluating on the test set...")
    truelabels, predictions = make_predictions(model, test_loader, device)

    # Compute and display accuracy
    accuracy = compute_accuracy(truelabels, predictions)
    print(f"Test Set Accuracy: {accuracy:.2f}%")

"""
# Training Loop
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Step [{i + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# Testing the Model
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


# Save model checkpoint after all epochs
save_path = 'model_checkpoint.pth'  # Path to save model checkpoints
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, save_path)"""