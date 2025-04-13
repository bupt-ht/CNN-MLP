import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.mlp import MLP
from models.cnn import CNN
from data_loader import load_cifar10
from config import config
import matplotlib.pyplot as plt

def train(model, trainloader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    
    # 记录到TensorBoard
    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_acc, epoch)
    
    return train_loss, train_acc

def test(model, testloader, criterion, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(testloader)
    test_acc = 100. * correct / total
    
    # 记录到TensorBoard
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', test_acc, epoch)
    
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
    
    return test_loss, test_acc

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'], help='model type')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='optimizer type')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    
    # 加载数据
    trainloader, testloader, classes = load_cifar10(args.batch_size)
    
    # 选择模型
    if args.model == 'mlp':
        model = MLP()
    else:
        model = CNN()
    model = model.to(config.device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=config.momentum, 
                             weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    
    # 创建保存目录
    save_dir = os.path.join(config.save_path, f"{args.model}_{args.optimizer}_lr{args.lr}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=save_dir)
    
    # 训练和测试
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"Training {args.model.upper()} with {args.optimizer.upper()} optimizer (lr={args.lr})")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch+1}/{args.epochs}")
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, epoch, writer)
        test_loss, test_acc = test(model, testloader, criterion, epoch, writer)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存模型
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    writer.close()

if __name__ == '__main__':
    main()
