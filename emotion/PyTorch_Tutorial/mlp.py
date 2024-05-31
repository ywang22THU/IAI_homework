import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision.ops import MLP

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


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
    model = MLP()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    print("Training...")
    for epoch in range(10):
        train(epoch, train_loader, model, criterion, optimizer)
        test(test_loader, model)
