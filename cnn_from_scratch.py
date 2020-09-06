import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

test_train_split = 0.2
np.random.seed(1000)


# dataset from image folders
data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
dataset = datasets.ImageFolder(root='./phase_wise_data', transform=data_transforms)

# creating random indices
indices = np.array(list(range(len(dataset))))
np.random.shuffle(indices)
split_at = int(test_train_split*len(dataset))
trains_ids, test_ids = indices[split_at:], indices[:split_at] 

# train and test sets
train_set = Subset(dataset, trains_ids)
test_set = Subset(dataset, test_ids)

train_batch_size = 32
train_loader = DataLoader(train_set, batch_size=train_batch_size)
test_loader = DataLoader(test_set, batch_size=len(test_set))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 4, stride=2)
        self.conv3 = nn.Conv2d(12, 18, 4, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(162, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, input):
        x = F.relu(self.pool(self.conv1(input)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = x.view(-1, 162)
        x = F.relu(self.fc1(x))       
        x = F.softmax(self.fc2(x))

        return x


model = CNN()
opt = optim.SGD(model.parameters(), lr = 0.01)
lossfn = nn.CrossEntropyLoss()
record_at = 10

def train(epochs):

    for i in range(epochs):

        for batch_idx, (data, labels) in enumerate(train_loader):
            out = model(data)
            opt.zero_grad()
            loss = lossfn(out, labels)
            loss.backward()
            opt.step()

            print('Epochs {}/{}   Iteration: {}/{}   Training Loss: {}'.format(i, epochs, batch_idx, \
                int(len(train_set)/train_batch_size), loss.item()))

            if batch_idx%record_at==0:
                torch.save(model.state_dict(), './model.pth')
                torch.save(opt.state_dict(), './optimizer.pth')


if __name__ == '__main__':
    train(128)





