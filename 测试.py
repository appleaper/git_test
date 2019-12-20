"""
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
"""
import torch
import torch.utils.data as Data
from dataloader import get_loader

BATCH_SIZE = 4

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
# 把数据放在数据库中
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

def show_batch():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))


if __name__ == '__main__':
    show_batch()