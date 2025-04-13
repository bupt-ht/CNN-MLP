import torch
class Config:
    # 数据配置
    data_path = './data'
    batch_size = 64
    num_workers = 4

    # 训练配置
    epochs = 50
    momentum = 0.9
    weight_decay = 5e-4

    # 模型保存路径
    save_path = './experiments'

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


config = Config()