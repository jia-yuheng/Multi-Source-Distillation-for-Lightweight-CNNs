
# run this file to generate resnet18_cifar100_best.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm

# ======================== 参数配置 ========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
num_classes = 100
warmup_epochs = 5
total_epochs = 200
mixup_epochs = 180
mixup_alpha = 1.0
log_dir = './runs/resnet18_cifar100_v2'
ckpt_path = './checkpoint/resnet18_cifar100_best.pth'
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

# ======================== 数据增强 ========================
class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

# ======================== 模型定义 ========================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

# ======================== 数据加载 ========================
def get_loaders():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader

# ======================== Mixup 实现 ========================
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ======================== 训练流程 ========================
def train():
    writer = SummaryWriter(log_dir=log_dir)
    model = ResNet18(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    train_loader, test_loader = get_loaders()
    best_acc = 0.0

    for epoch in range(1, total_epochs+1):
        model.train()
        total, correct, train_loss = 0, 0, 0.0

        # warmup 学习率
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * 0.1

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            if epoch <= mixup_epochs:
                inputs, targets_a, targets_b, lam = mixup_data(data, targets, mixup_alpha)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        scheduler.step()
        writer.add_scalar('Train/Loss', train_loss / total, epoch)
        writer.add_scalar('Train/Acc', acc, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)

        # 验证阶段
        model.eval()
        correct1, correct5, total = 0, 0, 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, pred = outputs.topk(5, 1, True, True)
                total += targets.size(0)
                correct1 += pred[:, 0].eq(targets).sum().item()
                correct5 += pred.eq(targets.view(-1, 1).expand_as(pred)).sum().item()

        top1 = 100. * correct1 / total
        top5 = 100. * correct5 / total
        writer.add_scalar('Test/Top1', top1, epoch)
        writer.add_scalar('Test/Top5', top5, epoch)
        print(f"Epoch [{epoch}/{total_epochs}] Train Acc: {acc:.2f}% | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")

        if top1 > best_acc:
            best_acc = top1
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ 新最佳模型已保存: Top-1 Acc {best_acc:.2f}%")

    writer.close()

if __name__ == '__main__':
    train()
