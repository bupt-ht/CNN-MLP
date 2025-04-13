import argparse
import torch
from models.mlp import MLP
from models.cnn import CNN
from data_loader import load_cifar10
from config import config

def test_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Testing')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'cnn'], help='model type')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
    args = parser.parse_args()
    
    # 加载数据
    _, testloader, _ = load_cifar10()
    
    # 加载模型
    if args.model == 'mlp':
        model = MLP()
    else:
        model = CNN()
    model = model.to(config.device)
    model.load_state_dict(torch.load(args.model_path))
    
    # 测试模型
    test_model(model, testloader)

if __name__ == '__main__':
    main()
EOL
}

# 生成工具脚本
generate_utils_script() {
    echo "生成工具脚本..."
    cat > utils.py <<EOL
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curves(train_losses, test_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_accuracy_curves(train_accs, test_accs, save_path=None):
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
