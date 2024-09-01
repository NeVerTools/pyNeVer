import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hdim, output_dim):
        super(SimpleNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hdim),  # First linear layer
            nn.ReLU(),  # ReLU activation
            nn.Linear(hdim, output_dim)  # Second linear layer (output layer)
        )

    def forward(self, x):
        return self.sequential(x)  # Forward pass through the sequential layers


# Example usage:

# Initialize model parameters
input_dim = 28 * 28  # For MNIST, images are 28x28
hdim = 128  # Hidden layer size
output_dim = 10  # Number of classes in MNIST

# Create the model
model = SimpleNN(input_dim, hdim, output_dim)

# Define a loss function and an optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load MNIST data
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=64,
                          shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=1000, shuffle=False)


# Training function
def train(model, device, train_loader, test_loader, optimizer, criterion, num_epochs):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

        test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


# Usage example
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, device, train_loader, test_loader, optimizer, criterion, num_epochs=10)
