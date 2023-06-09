import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import BasicNet

"""
Downloading datasets using pytorch default dataloader.
"""

#Download training data from open datasets.

training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download= True,
    transform=ToTensor(),
)

#Download test data from open datasets.
test_data = datasets.FashionMNIST(
 root = "data", 
 train=False,
 download=True,
 transform=ToTensor(),
 )

batch_size = 64

#Create pytorch dataloaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#checking data shape and type.
for X, y in test_dataloader:
    print(f"shape of X [N, C, H, W]: {X.shape}")
    print(f"shape of y: {y.shape} {y.dtype}")
    break

"""
Building a basic network model.
"""
#Getting cpu, gpu, or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device")

model = BasicNet().to(device)
print(model)


"""
Optimizing the model parameter.
"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Feedin the data to the model.
def train(dataloader, model, losses, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        #compute prediction error.
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss = loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= size
            print(f"test logs:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>.8f}%\n")

epochs = 5
for t in range(epochs):
    print(f"epoch {t+1}------------------------\n")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print("Done!")

#Saving the model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")