# Import libraries
from PIL import Image
import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Define model
train = MNIST(root='data', train=True, download=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)


# Image classifier class
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*22*22, 30),
        )
    def forward(self, x):
        return self.model(x)

# Define optimizer and loss function, initialize model
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


def exists(path):
    try:
        with open(path, 'rb'): pass
        return True
    except FileNotFoundError:
        return False

# Train model
if __name__ == "__main__":
    if not exists('model.pt'):
        for epoch in range(30):
            for batch in dataset:
                X, y = batch
                X, y = X.to('cuda'), y.to('cuda')
                y_pred = clf(X)
                loss = loss_fn(y_pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f"Epoch {epoch} loss: {loss.item()}")

        with open('model.pt', 'wb') as f:
            save(clf.state_dict(), f)

    with open('model.pt','rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_1.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    prediction = torch.argmax(clf(img_tensor))
    # Extract the prediction from output
    for index, char in enumerate(str(prediction)):
        if char.isdigit():
            print('The model thinks this number is:', char)
            break
    img = Image.open('img_5.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')
    for index, char in enumerate(str(prediction)):
        if char.isdigit():
            print('The model thinks this number is:', char)
            break

