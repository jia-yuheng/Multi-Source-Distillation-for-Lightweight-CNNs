
# Run to verify ResNet-18 teacher performance (81.48% on CIFAR-100).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ======================== 参数配置 ========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128
ckpt_path = './resnet18_cifar100_best.pth'

# CIFAR-100 类别名称（手动定义）
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# ======================== 模型结构复用 ========================
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
    def __init__(self, block, num_blocks, num_classes=100):
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

def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

# ======================== 测试集加载 ========================
def get_test_loader():
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

# ======================== 推理评估 ========================
def evaluate(model):
    test_loader = get_test_loader()
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
    print(f"✅ 推理完成：Top-1 Accuracy = {top1:.2f}%, Top-5 Accuracy = {top5:.2f}%")

# ======================== 单张图片推理 ========================
def predict_single_image(model, image_path):
    model.eval()
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    top5_prob = top5_prob.cpu().numpy()[0]
    top5_idx = top5_idx.cpu().numpy()[0]

    print("\n📷 单张图片预测结果：")
    for i in range(5):
        print(f"Top-{i+1}: {CIFAR100_CLASSES[top5_idx[i]]} ({top5_prob[i]*100:.2f}%)")

    # 可视化图像
    plt.imshow(np.array(image))
    plt.title("预测 Top-1: " + CIFAR100_CLASSES[top5_idx[0]])
    plt.axis('off')
    plt.show()

# ======================== 主函数 ========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    args = parser.parse_args()

    model = ResNet18().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    if args.image:
        predict_single_image(model, args.image)
    else:
        evaluate(model)
