import torch
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # Flatten data from (batch_size, 1, 28, 28) to (batch_size, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        return x

def train(epoch, train_loader, model, criterion, optimizer):
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f"[epoch {epoch}, batch {batch_idx}] loss: {running_loss / 300}")
            running_loss = 0.0


def test(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_loader:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f"Accuracy on test set: {100 * correct / total}%%")


if __name__ == '__main__':
    print("Loading data...")
    train_dataset = datasets.MNIST(root='./dataset/minist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(root='./dataset/minist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

    print("Building model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    print("Training...")
    for epoch in range(10):
        train(epoch, train_loader, model, criterion, optimizer)
        test(test_loader, model)

