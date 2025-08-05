import torch
import torch.nn as nn
import torch.nn.functional as F

# 小模型A：细粒度边缘
class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
    def forward(self, x):
        out = F.relu(self.conv(x))
        res = out - x
        return res

# 小模型B：粗粒度纹理
class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=5, padding=2)
    def forward(self, x):
        out = F.relu(self.conv(x))
        res = out - x
        return res

# 小模型C：池化强化平移不变性
class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        out = self.pool(F.relu(self.conv(x)))
        res = F.interpolate(out, size=x.shape[2:]) - x
        return res

# 小模型D：空洞卷积扩大感受野
class ModelD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2)
    def forward(self, x):
        out = F.relu(self.conv(x))
        res = out - x
        return res

# 大模型：整合残差特征
class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 10, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x

# 整合流程
class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modelA = ModelA()
        self.modelB = ModelB()
        self.modelC = ModelC()
        self.modelD = ModelD()
        self.bigModel = BigModel()
    def forward(self, x):
        resA = self.modelA(x)
        resB = self.modelB(x)
        resC = self.modelC(x)
        resD = self.modelD(x)
        # 拼接残差特征
        features = torch.cat([resA, resB, resC, resD], dim=1)
        out = self.bigModel(features)
        return out

# 用法示例
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载MNIST数据集
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model = EnsembleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    lamuda = 1e-4  # 初始正则系数

    # 训练与验证
    for epoch in range(10):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            logits = model(x)
            ce_loss = criterion(logits, y)
            l2_reg = sum((param**2).sum() for param in model.parameters())
            loss = ce_loss + lamuda * l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                ce_loss = criterion(logits, y)
                l2_reg = sum((param**2).sum() for param in model.parameters())
                loss = ce_loss + lamuda * l2_reg
                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        # 动态调整lamuda
        if epoch > 0 and prev_val_loss - val_loss < 0.001:
            lamuda *= 1.1  # 增大正则
        else:
            lamuda *= 0.95  # 减小正则
        prev_val_loss = val_loss

        scheduler.step()
        print(f"Epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, lamuda: {lamuda:.6f}, lr: {optimizer.param_groups[0]['lr']}")

    # 测试集评估
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")