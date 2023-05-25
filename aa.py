import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 장치 설정 (GPU를 사용할 수 있다면 GPU로 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 불러오기 및 전처리
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 신경망 모델 정의
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 입력 레이어 -> 숨겨진(hidden) 레이어
        self.fc2 = nn.Linear(128, 10)  # 숨겨진(hidden) 레이어 -> 출력 레이어

    def forward(self, x):
        x = x.view(-1, 784)  # 입력 이미지를 1차원으로 변환
        x = torch.relu(self.fc1(x))  # 숨겨진(hidden) 레이어에 ReLU 활성화 함수 적용
        x = self.fc2(x)  # 출력 레이어에 적용할 활성화 함수는 손실 함수에서 처리
        return x

model = NeuralNet().to(device)

# 손실 함수 및 최적화 기준 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 학습 함수 정의
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 테스트 함수 정의
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
train(train_dataset)