import torch


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == '__main__':
    x_data = torch.Tensor([[1.0], [2.0], [3.0]])
    y_data = torch.Tensor([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    model = LinearModel()

    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print("epoch:", epoch, "loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("w:", model.linear.weight.detach().numpy())
    print("b:", model.linear.bias.detach().numpy())
    y_test = model(torch.Tensor([4.0]))
    print("y_test:", y_test.detach().numpy())
