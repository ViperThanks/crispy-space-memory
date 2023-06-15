import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=5,
                                         shuffle=False, num_workers=2)
classes = ('airplane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize1
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(5)))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01, weight_decay=5e-4) 




for epoch in range(70):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10000 == 9999:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 10000) )#打印每10000张图片后的loss值
            running_loss = 0.0

print('训练完成！')



dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
outputs = net(images)
predicted = torch.max(outputs, 1)
print(' '.join('%5s' % classes[labels[j]] for j in range(5)))

tr_correct = 0
tr_total = 0
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        tr_total += labels.size(0)
        tr_correct += (predicted == labels).sum().item()

print('50000张图片训练集的准确率: %d %%' % (100 * tr_correct / tr_total))

train_correct = list(0. for i in range(10))
train_total = list(0. for i in range(10))
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        outputs = net(images)
        _,predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            train_correct[label] += c[i].data.item()
            train_total[label] += 1

for i in range(10):
    print(' %5s 准确率 : %2d %%' % (
        classes[i], 100 * train_correct[i] / train_total[i]))

te_correct = 0
te_total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        te_total += labels.size(0)
        te_correct += (predicted == labels).sum().item()

print('10000张图片测试集的准确率: %d %%' % (100 * te_correct /te_total))



test_correct = list(0. for i in range(10))
test_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _,predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            test_correct[label] += c[i].data.item()
            test_total[label] += 1

for i in range(10):
    print('%5s 准确率 : %2d %%' % (
        classes[i], 100 * test_correct[i] / test_total[i]))


# 计算每个类别的准确率
te_accuracies = []
for i in range(10):
    accuracy = 100 * test_correct[i] / test_total[i]
    te_accuracies.append(accuracy)
tr_accuracies = []
for i in range(10):
    accuracy = 100 * train_correct[i] / train_total[i]
    tr_accuracies.append(accuracy)

# 绘制准确率曲线图
plt.plot(classes, tr_accuracies, label='train')
plt.plot(classes, te_accuracies, label='test')

plt.ylabel('Accuracies(%)')
plt.xlabel('Classes')
plt.ylim(0, 100)
plt.xticks(classes, classes, rotation=0)
plt.legend()
for i, acc in enumerate(te_accuracies):
    plt.text(classes[i], acc -3, f"{acc:.1f}", ha='left')
for i, acc in enumerate(tr_accuracies):
    plt.text(classes[i], acc -3, f"{acc:.1f}", ha='left')

plt.show()