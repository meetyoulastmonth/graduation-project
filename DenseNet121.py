import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# 定义数据变换
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

# 获取数据
train_data = torchvision.datasets.ImageFolder(root='./data/train',
                                              transform=transform)
test_data = torchvision.datasets.ImageFolder(root='./data/test',
                                             transform=transform)

# 加载数据
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4,
                                          shuffle=False)

# 定义mobilenet网络
model = torchvision.models.densenet121(weights=None, num_classes=32)


# 将网络架构修改为32分类
# model.classifier[1] = torch.nn.Linear(1280, 32)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

loss_list=[]
epochs = 100
training_loss = 0.0
accuracies = []
max_accuracy = 0.0
for epoch in range(epochs):
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    torch.cuda.empty_cache()
    training_loss = running_loss / total
    loss_list.append(training_loss)
    accuracy = correct / total
    accuracies.append(accuracy)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
    print('[Epoch %d] Loss: %.3f Acc: %.3f' %
          (epoch + 1, training_loss, accuracy))


# 测试模型
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy:', correct / total)


# 保存权重并绘制训练曲线
torch.save(model.state_dict(), 'Model_mobilenet.pth')

x_ori = []
for i in range(len(loss_list)):
    x_ori.append(i + 1)
plt.title("Epoch-Loss")
plt.plot(x_ori, loss_list)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

x_ori = []
for i in range(len(accuracies)):
    x_ori.append(i + 1)
plt.title("Epoch-Accuracy")
plt.plot(x_ori, accuracies)
plt.ylabel("acc")
plt.xlabel("epoch")
plt.show()