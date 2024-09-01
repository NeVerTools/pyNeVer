import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.onnx

# Parametri
epochs = 100
train_batch_size = 128
test_batch_size = 64
learning_rate = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"


# Trasformazione per il dataset MNIST, includendo il flatten
transform = transforms.Compose([
    transforms.ToTensor(),  # Converti l'immagine in un tensore
    transforms.Lambda(lambda x: x.view(-1))  # Applica il flatten per trasformare l'immagine in un vettore 1D
])

# Caricamento del dataset MNIST
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Suddivisione del dataset in set di addestramento e di test
train_size = int(len(full_dataset) * 0.8)  # Usa 80% per l'addestramento e 20% per il test
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

# Definizione delle architetture delle reti neurali senza Batch Normalization
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)  # Uscita con 10 unità per 10 classi

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)  # Uscita con 10 unità per 10 classi

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)  # Uscita con 10 unità per 10 classi

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train(model, train_loader, criterion, optimizer, scheduler, epochs=5):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Scheduler step
        scheduler.step()

        # Print training status
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def test_and_save_accuracy(model, test_loader, accuracy_file):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    # Salva l'accuratezza su un file di testo
    with open(accuracy_file, 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n')

def save_model_onnx(model, file_name):
    model.to(device)
    dummy_input = torch.randn(1, 28 * 28).to(device)  # Input dummy con dimensioni corrispondenti a MNIST
    torch.onnx.export(model, dummy_input, file_name, input_names=['input'], output_names=['output'])
    print(f'Model saved in ONNX format as {file_name}')


# Seleziona la rete neurale da utilizzare
model = Net3()  # Cambia in Net1() o Net2() per altre architetture

# Definizione della funzione di perdita, dell'ottimizzatore e dello scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Addestramento della rete
train(model, train_loader, criterion, optimizer, scheduler, epochs=epochs)

# Testing della rete e salvataggio dell'accuratezza
accuracy_file = 'accuracy.txt'
test_and_save_accuracy(model, test_loader, accuracy_file)

# Salvataggio del modello in formato ONNX
onnx_file = 'model.onnx'
save_model_onnx(model, onnx_file)
