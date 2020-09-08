import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

test_train_split = 0.2
np.random.seed(100)


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
        self.conv3_drop = nn.Dropout2d()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(162, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, input):
        x = F.relu(self.pool(self.conv1(input)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))
        x = x.view(-1, 162)
        x = F.relu(self.fc1(x))   
        x = F.dropout(x, training=self.training)    
        x = self.fc2(x)       

        return x


model = CNN()
# model.load_state_dict(torch.load('./model.pth'))
opt = optim.SGD(model.parameters(), lr = 0.01)
lossfn = nn.CrossEntropyLoss()
record_at = 10

epochs = 512
train_losses = []
train_count = 0
def train(epoch):
    global train_count
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        out = model(data)
        opt.zero_grad()
        loss = lossfn(out, labels)
        loss.backward()
        opt.step()
        train_losses.append(loss.item())
        train_count += 1

        print('Epochs {}/{}   Iteration: {}/{}   Training Loss: {:.4f}'.format(epoch, epochs, batch_idx, \
            int(len(train_set)/train_batch_size), loss.item()))

        if batch_idx%record_at==0:
            torch.save(model.state_dict(), './model.pth')
            torch.save(opt.state_dict(), './optimizer.pth')

top1 = [0, 0]
test_losses = []
def test(epoch):
    model.eval()
    loss = 0 
    correct = 0
    print("=========== testing ============")
    with torch.no_grad():
        for _, (data, label) in enumerate(test_loader):
            out = model(data)
            loss += F.cross_entropy(out, label)
            _, output = torch.max(out, 1)
            correct += (output == label).float().sum()

    avg_loss = loss/len(test_set)
    accuracy = correct/len(test_set)
    if accuracy*100 > top1[1]:
        top1[0] = epoch
        top1[1] = accuracy*100
    test_losses.append(avg_loss)
    print("Avg. test loss: {:.4f}   Accuracy: {}/{}  ({:.4f}%)   top1: {:.4f} at {}".format(avg_loss, correct, len(test_set), 100*accuracy, top1[1], top1[0]))
    
def plot_results():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Best performance: {:.4f} at {}th epoch'.format(top1[1], top1[0]))
    ax1.plot([x for x in range(1, train_count+1)], [loss for loss in train_losses])
    ax2.plot([epoch for epoch in range(1, epochs+1)], [loss for loss in test_losses])
    ax1.set_xlabel('data points')
    ax2.set_xlabel('epochs')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test loss')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




if __name__ == '__main__':
    for epoch in range(1, epochs+1):
        train(epoch)
        test(epoch)
    plot_results()






