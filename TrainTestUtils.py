import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

def compute_accuracy(truelabels, predictions):
    """
    Computes the accuracy of predictions.

    Args:
        truelabels (list): List of true labels.
        predictions (list): List of predicted labels.

    Returns:
        float: Accuracy as a percentage.
    """
    correct = sum(t == p for t, p in zip(truelabels, predictions))
    total = len(truelabels)
    accuracy = 100.0 * correct / total
    return accuracy

def train(model, device, loss_criteria, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)

    # Wrap the training loop with tqdm for the progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    # Process the images in batches
    for batch_idx, (data, target) in progress_bar:
        # Use the CPU or GPU as appropriate
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

    # return average loss for the epoch
    avg_loss = train_loss / len(train_loader)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def validate(model, device, loss_criteria, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss


def save_model(model, path):
    """
    Saves the trained model to the specified path.

    Args:
        model (torch.nn.Module): The trained PyTorch model to save.
        path (str): The file path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def make_predictions(model, data_loader, device):
    """
    Make predictions using the trained model on a given data loader.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): The data loader for the dataset.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: (list of true labels, list of predictions)
    """
    model.eval()
    truelabels = []
    predictions = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            # Append true labels
            truelabels.extend(target.cpu().numpy())

            # Get predictions
            output = model(data)
            predicted = output.argmax(1).cpu().numpy()
            predictions.extend(predicted)

    return truelabels, predictions

def plot_loss(epoch_nums, training_loss, validation_loss):
    """
    Plots the training and validation loss over epochs.

    Args:
        epoch_nums (list): List of epoch numbers.
        training_loss (list): List of training losses.
        validation_loss (list): List of validation losses.
    """
    plt.figure(figsize=(15, 15))
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
