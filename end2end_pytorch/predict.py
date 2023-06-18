import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import ToTensor
from model import BasicNet

#downloading the test dataset
test_data = datasets.FashionMNIST(
 root = "data", 
 train=False,
 download=True,
 transform=ToTensor(),
 )

#Getting cpu, gpu, or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() 
    else "cpu"
)

#Loading the saving model
model = BasicNet().to(device)
model.load_state_dict(torch.load('model.pth'))

#Now, making the prediction
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')