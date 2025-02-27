import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from CustomCNN import CustomCNN
from DataLoader import Stanford40Dataset
from TrainTestUtils import train, validate, make_predictions, plot_loss, save_model, compute_accuracy

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
EPOCHS = 3
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
    model = CustomCNN(num_classes=40).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_criteria = nn.CrossEntropyLoss()

    # Track metrics in these arrays
    epoch_nums = []
    training_loss = []
    validation_loss = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, device, loss_criteria, train_loader, optimizer, epoch)
        test_loss = validate(model, device, loss_criteria, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

    plot_loss(epoch_nums, training_loss, validation_loss)
    save_model(model, "model")

    print("Getting predictions from test set...")
    truelabels, predictions = make_predictions(model, test_loader, device)

    accuracy = compute_accuracy(truelabels, predictions)
    print(f"Test Set Accuracy: {accuracy:.2f}%")

