import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import save

# Set device to use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a series of image transforms for data augmentation and normalization.
num_epoch = 100
batch_size = 3

# Define ImageFolder datasets for training and testing sets
train_set = torchvision.datasets.ImageFolder(root="/home/nikodem/ANN/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.ImageFolder(root="/home/nikodem/ANN/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

# Classes based on images
classes = ('paper', 'rock', 'scissors')


# Define ImageFolder datasets for the training and testing sets, and load them into PyTorch's DataLoader, 
# which is used for batching and shuffling the data during training.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv3 = nn.Conv2d(10, 14, 5)
        self.conv4 = nn.Conv2d(14, 20, 5)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 3)

    def forward(self, x):
        #print(x.shape)

        #x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)

        #x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)

        #x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv3(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)

        #x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv4(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)

        #x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = x.view(-1, 20*10*10)
        #print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 3 == 00:    # print every 2000 mini-batches
            print (f'Epoch [{epoch+1}/{num_epoch}],     Step [{i+1}/{len(train_loader)}],   Loss: {loss.item():.4f}')
            running_loss = 0.0

print('Finished Training')


PATH = './model.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            # if i >= len(labels):                #????????????????????????????????????????????????
            #     break
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')