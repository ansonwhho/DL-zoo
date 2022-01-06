"""
Implementation of a simple CNN,
based on Yann LeCunn's original paper

http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf

Anson Ho and Ole Jorgensen, 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    """
    Load and preprocess MNIST data
    """

    # Load datasets
    img_transforms = transforms.Compose([transforms.ToTensor()])
    train_val_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=img_transforms)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=img_transforms)

    # Split into train and validation sets
    n = len(train_val_set)
    split = int(config["train_ratio"] * n)
    train_set, val_set = D.random_split(train_val_set, [split, n - split])

    train_loader = D.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = D.DataLoader(val_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = D.DataLoader(test_set, batch_size=config["batch_size"], shuffle=True)

    return train_loader, val_loader, test_loader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,12,5)
        self.conv2 = nn.Conv2d(12,12,4)
        self.layer3 = nn.Linear(192,30)
        self.layer4 = nn.Linear(30,10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = torch.flatten(x,1)
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def optimise(optimiser, inputs, labels):
    """
    Implements a single optimisation
    step within a training loop

    Args (for MNIST dataset)
        optimiser: the optimisation algorithm of choice, e.g. SGD
        inputs: a tensor of images
        labels: a tensor of indices
    """

    optimiser.zero_grad()
    outputs = model(inputs.float())

    # use one-hot encoding for labels to compute loss
    OHE_labels = F.one_hot(labels, num_classes=10).float()
    loss = loss_function(outputs, OHE_labels)
    loss.backward()
    optimiser.step()

    return loss.item()

def train(dataloader):
    """
    Training loop
    Saves model in specified path
    """

    train_loss = 0

    for epoch in range(config["num_epochs"]):
        print(f"EPOCH NUMBER {epoch+1} OF {config['num_epochs']}")
        for idx, data in enumerate(dataloader, start=0):
            images, labels = data
            train_loss += optimise(optimiser, images, labels)

            if (idx+1) % config["batches_per_eval"] == 0:
                print(f"Training loss at epoch {epoch+1}, batch {idx+1} is {train_loss/config['batches_per_eval']:.6f}")
                train_loss = 0
    
    print("Finished training")
    torch.save(model.state_dict(), config["model_path"])

def test(dataloader):
    """
    Determine the test set accuracy
    of a model
    """

    correct = 0
    total = 0

    for data in dataloader:
        images, labels = data
        output = model(images)
        prediction = torch.argmax(output, dim=1)

        total += labels.size(0)
        correct += (prediction == labels).sum().item()
    
    score = 100 * correct / total
    
    return score

def sample_img(dataloader):
    """
    Returns an image as an
    numpy array
    """
    images, labels = next(iter(dataloader))
    output = model(images)
    prediction = torch.argmax(output, dim=1)[0]

    # Select single image
    img = images[0].squeeze()
    label = labels[0]
    np_img = np.array(img)
    plt.imshow(np_img)
    plt.show()

    print(f"The true value is {label}")
    print(f"The predicted value is {prediction}")

if __name__ == "__main__":

    TRAIN = 0 # 0 to test performance, 1 to train model
    SAMPLE_IMG = 1 # Show sample prediction

    config = {
        "learn_rate": 1e-1,
        "num_epochs": 4,
        "train_ratio": 0.95,
        "batch_size": 8,
        "batches_per_eval": 1000,
        "model_path": "./models/CNN.pth"
    }

    train_loader, val_loader, test_loader = get_data()

    if TRAIN:
        model = Net()
        loss_function = nn.MSELoss()
        optimiser = optim.SGD(model.parameters(), lr=config["learn_rate"])
        train(train_loader)
        score = test(val_loader)
        print(f"Validation set accuracy is {score}")

    else: 
        model = Net()
        model.load_state_dict(torch.load(config["model_path"]))
        score = test(test_loader)
        print(f"Test set accuracy is {score}")

    if SAMPLE_IMG:
        sample_img(test_loader)