import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Define the neural network class
class CustomNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, filters_number: int, kernel_size: int, stride: int,
                 padding: int, hidden_layer_dim: int):
        super(CustomNN, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(1, filters_number, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(filters_number)  # Batch normalization for conv layer
        self.flatten = nn.Flatten()

        # Calculate input size for fc1 dynamically
        dummy_input = torch.zeros(1, 1, input_dim, input_dim)
        conv_output = self.conv(dummy_input)
        conv_output = self.bn1(conv_output)  # Ensure BatchNorm compatibility
        conv_output_flatten = self.flatten(conv_output)
        fc1_in_features = conv_output_flatten.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_in_features, hidden_layer_dim)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_layer_dim, output_dim)

    def forward(self, x):
        # Apply convolution, batch norm, and ReLU
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1_dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Hyperparameters
input_dim = 28  # Assuming MNIST dataset
output_dim = 10
filters_number = 16
kernel_size = 3
stride = 1
padding = 1
hidden_layer_dim = 50
batch_size = 128
test_batch_size = 64
epochs = 50
learning_rate = 0.001

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
subset_indices = torch.randperm(len(train_dataset))[:5000]  # Randomly select 4000 samples
train_dataset = Subset(train_dataset, subset_indices)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNN(input_dim, output_dim, filters_number, kernel_size, stride, padding, hidden_layer_dim).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # Move data to device if using GPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
