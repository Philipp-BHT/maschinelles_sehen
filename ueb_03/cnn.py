# source code inspireed by
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CATEGORIES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


# implement your own NNs
class MyNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.kernel = 3
        self.stride = 1

        # First Conv Block (input size: 28x28)
        self.pad1 = self.calc_same_padding(kernel=self.kernel, stride=self.stride, input_size=28, output_size=28)
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=self.kernel, padding=self.pad1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=self.kernel, padding=self.pad1)
        self.pool1 = nn.MaxPool2d(2, 2)  # → 14x14
        self.dropout1 = nn.Dropout(0.25)

        # Second Conv Block (input size: 14x14)
        self.pad2 = self.calc_same_padding(kernel=self.kernel, stride=self.stride, input_size=14, output_size=14)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=self.kernel, padding=self.pad2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=self.kernel, padding=self.pad2)
        self.pool2 = nn.MaxPool2d(2, 2)  # → 7x7
        self.dropout2 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    @staticmethod
    def calc_same_padding(kernel, stride, input_size, output_size):
        pad_total = int((stride * (output_size - 1) - input_size + kernel))
        pad = pad_total // 2
        return pad

    def name(self):
        return "MyNeuralNetwork"

class AlexNet(MyNeuralNetwork):
    """
    Inspired by AlexNet, but channel number, padding and stride adapted from the MyNeuralNetwork example
    """
    def __init__(self):
        super(AlexNet, self).__init__()
        self.kernel1 = 11
        self.kernel2 = 5
        self.kernel3 = 3
        self.pad3 = self.calc_same_padding(self.kernel3, self.stride, 6, 6)

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, padding=self.pad1)
        self.pool1 = nn.MaxPool2d(self.kernel1, self.stride)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=self.kernel2, padding=self.pad2)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=self.kernel3, padding=self.pad3)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.pool1(x)
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "MyAlexNet"


class VGG16(MyNeuralNetwork):
    def __init__(self):
        super(VGG16, self).__init__()

        self.kernel = 3
        self.pad = 1
        self.stride = 1

        # Block 1: 2 conv layers
        self.block1 = nn.ModuleList([
            nn.Conv2d(1, 64, kernel_size=self.kernel, padding=self.pad),
            nn.Conv2d(64, 64, kernel_size=self.kernel, padding=self.pad)
        ])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: 3 conv layers
        self.block2 = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=self.kernel, padding=self.pad),
            nn.Conv2d(128, 128, kernel_size=self.kernel, padding=self.pad),
            nn.Conv2d(128, 128, kernel_size=self.kernel, padding=self.pad)
        ])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1 (2x conv + 1 pool)
        for conv in self.block1:
            x = F.relu(conv(x))
        x = self.pool1(x)

        # Block 2 (3x conv + 1 pool)
        for conv in self.block2:
            x = F.relu(conv(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "MyVGG16"

def training(model, data_loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for batch_idx, (inputs, labels) in enumerate(data_loader):

        # zero the parameter gradients
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # backward
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if batch_idx % 10 == 0:
            print(f'Training Batch: {batch_idx:4} of {len(data_loader)}')

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def test(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    # do not compute gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if batch_idx % 10 == 0:
                print(f'Test Batch: {batch_idx:4} of {len(data_loader)}')

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

    print('-' * 10)
    print(f'Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return epoch_loss, epoch_acc


def plot(train_history, test_history, metric, num_epochs, model):

    plt.title(f"Validation/Test {metric} vs. Number of Training Epochs")
    plt.xlabel(f"Training Epochs")
    plt.ylabel(f"Validation/Test {metric}")
    plt.plot(range(1, num_epochs + 1), train_history, label="Train")
    plt.plot(range(1, num_epochs + 1), test_history, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.savefig(f"{metric}_{model.name}.png")
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(0)

# hyperparameter
# TODO: find good hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001
momentum = 0.8

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])

# load train and test data
root = './data'
train_set = datasets.FashionMNIST(root=root,
                                  train=True,
                                  transform=transform,
                                  download=True)
test_set = datasets.FashionMNIST(root=root,
                                 train=False,
                                 transform=transform,
                                 download=True)

loader_params = {
    'batch_size': batch_size,
    'num_workers': 0  # increase this value to use multiprocess data loading
}

train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_params)
test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_params)

## model setup

def train_model(model: MyNeuralNetwork):
    model = model.to(device)
    print(f"Training Model {model.name()}")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_acc_history = []
    test_acc_history = []

    train_loss_history = []
    test_loss_history = []

    best_acc = 0.0
    since = time.time()

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # train
        training_loss, training_acc = training(model, train_loader, optimizer,
                                               criterion, device)
        train_loss_history.append(training_loss)
        train_acc_history.append(training_acc)

        # test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        test_loss_history.append(test_loss)
        test_acc_history.append(test_acc)

        # overall best model
        if test_acc > best_acc:
            best_acc = test_acc
            #  best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(
        f'Training complete in {(time_elapsed // 60):.0f}m {(time_elapsed % 60):.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')

    # plot loss and accuracy curves
    train_acc_history = [h.cpu().numpy() for h in train_acc_history]
    test_acc_history = [h.cpu().numpy() for h in test_acc_history]

    plot(train_acc_history, test_acc_history, 'accuracy', num_epochs, model)
    plot(train_loss_history, test_loss_history, 'loss', num_epochs, model)

    # plot examples
    example_data, _ = next(iter(test_loader))
    with torch.no_grad():
        output = model(example_data)

        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Pred: {}".format(CATEGORIES[output.data.max(
                1, keepdim=True)[1][i].item()]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig(f"examples_{model.name}.png")
        plt.show()

model_1 = MyNeuralNetwork()
model_2 = AlexNet()
model_3 = VGG16()
train_model(model_3)

