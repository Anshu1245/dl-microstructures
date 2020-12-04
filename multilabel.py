import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

test_train_split = 0.1
np.random.seed(500)


# dataset from image folders
data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
# transforms.Grayscale(num_output_channels=1), 
dataset1 = datasets.ImageFolder(root='./class_wise_data', transform=data_transforms)
dataset2 = datasets.ImageFolder(root='./phase_wise_data', transform=data_transforms)

# creating random indices
indices = np.array(list(range(len(dataset1))))
np.random.shuffle(indices)
split_at = int(test_train_split*len(dataset1))
trains_ids, test_ids = indices[split_at:], indices[:split_at] 

# train and test sets
train_set1 = Subset(dataset1, trains_ids)
test_set1 = Subset(dataset1, test_ids)

indices = np.array(list(range(len(dataset2))))
np.random.shuffle(indices)
split_at = int(test_train_split*len(dataset2))
trains_ids, test_ids = indices[split_at:], indices[:split_at] 

# train and test sets
train_set2 = Subset(dataset2, trains_ids)
test_set2 = Subset(dataset2, test_ids)


train_batch_size = 32
test_batch_size = 64
train_loader1 = DataLoader(train_set1, batch_size=train_batch_size)
test_loader1 = DataLoader(test_set1, batch_size=test_batch_size)
train_loader2 = DataLoader(train_set2, batch_size=train_batch_size)
test_loader2 = DataLoader(test_set2, batch_size=test_batch_size)


class CNN(nn.Module):
    
    def __init__(self, pretrained=True, dataset_name=None):
        super(CNN, self).__init__()

        self.network = models.resnet50(pretrained=True)
        
        # Freeze all layers except classifier and all but the last layer.
        for param in self.network.parameters():
            param.requires_grad = False

        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features=in_features, out_features=100)
        self.phase = nn.Linear(100, 10)
        self.c = nn.Linear(100, 7)
            
        
        '''
        for param in self.network.layer4.parameters():
            param.requires_grad = True 
        '''
        
    def forward(self, inputs):
        x=self.network(inputs)
        return self.c(x), self.phase(x) 


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = CNN()
#model = model.to(device)
opt = optim.SGD(model.parameters(), lr = 0.001)
lossfn = nn.CrossEntropyLoss()

epochs = 1024
train_losses = []
train_count = 0


def train(epoch):
    print("training")
    global train_count
    model.train()
    for batch_idx, (data1, data2) in enumerate(zip(train_loader1, train_loader2)):
        x1,l1=data1
        x1=x1.to(device)
        l1=l1.to(device)
        x2,l2=data2
        x2=x2.to(device)
        l2=l2.to(device)
        out1, _ = model(x1)
        _, out2 = model(x2)
        opt.zero_grad()
        loss1 = lossfn(out1, l1)
        loss2 = lossfn(out2, l2)
        loss=loss1+loss2
        loss.backward()
        opt.step()
        train_losses.append(loss.item())
        train_count += 1

        print('Epochs {}/{}   Iteration: {}/{}   Training Loss: {:.4f}'.format(epoch, epochs, batch_idx, int(len(train_set1)/train_batch_size), loss.item()))


top1 = [0, 0]
test_losses = []
def test(epoch, mode):
    model.eval()
    loss = 0 
    correct = 0
    print("=========== testing ============")
    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(zip(test_loader1, test_loader2)):
            x1,l1=data1
            x1=x1.to(device)
            l1=l1.to(device)
            x2,l2=data2
            x2=x2.to(device)
            l2=l2.to(device)
            out1, g = model(x1)
            g, out2 = model(x2)
            loss += F.cross_entropy(out1, l1)
            loss += F.cross_entropy(out2, l2)
            g, out1 = torch.max(out1, 1)
            g, out2 = torch.max(out2, 1)
            correct += (out1 == l1).float().sum()
            correct += (out2 == l2).float().sum()
    
    avg_loss = loss/(2*len(test_set1))
    accuracy = correct/(2*len(test_set1))
    if accuracy*100 > top1[1]:
        top1[0] = epoch
        top1[1] = accuracy*100
        if mode:
            save_progress(epoch)
    test_losses.append(avg_loss)
    print("Avg. test loss: {:.5f}   Accuracy: {}/{}  ({:.4f}%)   top1: {:.4f} at {}".format(avg_loss, correct, 2*len(test_set1), 100*accuracy, top1[1], top1[0]))

def save_progress(epoch):
    print("======saving model======")
    torch.save({'model':model.state_dict(),
                'optimizer':opt.state_dict()},
                './saved_model/{}.tar'.format(epoch))

def load_saved():
    print("======loading saved model======")
    state = torch.load('./saved_model/257.tar') #TODO
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    return model, opt

    
def plot_results():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Best performance: {:.4f} at {}th epoch'.format(top1[1], top1[0]))
    ax1.plot([x for x in range(1, train_count+1)], [loss for loss in train_losses])
    ax2.plot([epoch for epoch in range(1, epochs+1)], [loss for loss in test_losses])
    ax1.set_xlabel('batches')
    ax2.set_xlabel('epochs')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test loss')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



mode = {'train':1, 'infer':0}
if __name__ == '__main__':
    model, opt = load_saved()
    model = model.to(device)
    for epoch in range(258, epochs+1):
        train(epoch)
        test(epoch, mode['train'])
    plot_results()
    
    
    
    '''
    test(0, mode['infer'])
    '''







