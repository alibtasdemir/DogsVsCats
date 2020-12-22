import torch
from torch import nn
from torch.nn import functional as F
import config

class DogCatModel(nn.Module):
    def __init__(self, num_classes):
        super(DogCatModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear((config.IMG_HEIGHT//8)*(config.IMG_WIDTH//8)*128, 128)
        self.drop_1 = nn.Dropout(0.2)

        self.output = nn.Linear(128, num_classes)

    def forward(self, images, targets=None):
        bs, c, h, w = images.size()     # [1, 3, 128, 128]
        #print(images.size())
        x = F.relu(self.conv1(images))
        x = self.max_pool_1(x)
        #print(x.size())                 # [1, 32, 64, 64]
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        #print(x.size())                 # [1, 64, 32, 32]
        x = F.relu(self.conv3(x))
        x = self.max_pool_3(x)
        #print(x.size())                 # [1, 128, 16, 16]
        x = x.view(bs, -1)
        x = self.linear_1(x)
        x = self.drop_1(x)
        #print(x.size())                 # [1, 128]
        x = self.output(x)
        #print(x.size())                 # [1, 2]
        if targets is not None:
            #print("Loss")
            #print(x.size())
            #print(targets.size())
            #target = torch.max(targets, 1)[1]
            #print(target.size())
            criterion = nn.CrossEntropyLoss()
            loss = criterion(x, targets)
            return x, loss

        return x, None

if __name__ == "__main__":
    cm = DogCatModel(2)
    img = torch.rand(5, 3, 128, 128)
    target = torch.randint(0,1, (5,1))
    x, loss = cm(img, target)
    print(x)
    print(loss)