import numpy as np
import onnx
import onnxsim
import scipy.io as sio
import visdom
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import time


class SVHN(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.file_path = file_path
        data_mat = sio.loadmat(self.file_path)
        self.X = data_mat["X"]
        self.y = data_mat["y"]

    def __getitem__(self, index) -> T_co:
        return self.X[:, :, :, index], self.y[index]

    def __len__(self):
        return self.y.shape[0]


from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = models.resnet18()
# 修改全连接层的输出
num_ftrs = resnet18.fc.in_features
# 十分类，将输出层修改成10
resnet18.fc = nn.Linear(num_ftrs, 10)
# 模型参数放大GPU上，加快训练速度
resnet18 = resnet18.to(device)

def train(model, dataLoader, optimizer, lossFunc, n_epoch):
    start_time = time.time()
    test_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    total_batch = 0  # 记录进行到多少batch
    writer = SummaryWriter(log_dir=log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(n_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, n_epoch))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for batch_idx, dataset in enumerate(dataLoader):
            length = len(dataLoader)
            optimizer.zero_grad()
            data, labelOrg = dataset
            data = data.to(device)
            label = F.one_hot(labelOrg.to(torch.long), 10).to(torch.float).to(device)
            predict = model(data)
            loss = lossFunc(predict, labelOrg)
            # loss = lossFunc(predict, label)
            loss.backward()
            optimizer.step()
            # Tensor.item() 类型转换，返回一个数
            sum_loss += loss.item()
            # maxIdx, maxVal = torch.max
            _, predicted = torch.max(predict.data, dim=1)
            total += label.size(0)
            correct += predicted.cpu().eq(labelOrg.data).sum().item()
            # 注意这里是以一个batch为一个单位
            print("[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% "
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1), 100. * correct / total))
            # 每一百个batch计算模型再测试集或者验证集的正确率
            if total_batch % 100 == 0:
                testDataLoss, testDataAcc = evalTestAcc(model)
                time_dif = get_time_dif(start_time)
                if testDataLoss < test_best_loss:
                    test_best_loss = testDataLoss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Test Loss: {3:>5.2},  Test Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, sum_loss / (batch_idx + 1), correct / total, testDataLoss, testDataAcc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", testDataLoss, total_batch)
                writer.add_scalar("acc/train", correct / total, total_batch)
                writer.add_scalar("acc/dev", testDataAcc, total_batch)
            # 提供训练程序的两个出口： n_epoch， require_improvement个batch没有提升
            total_batch += 1
            model.train()
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()

def get_time_dif(start_time):
    end_time = time.time()
    return (start_time - end_time)

def evalTestAcc(net):
    net.eval()
    totalAcc = 0.0
    sumLoss = 0.0
    total = 0.0
    with torch.no_grad():
        for idx, dataset in enumerate(testDataLoader):
            data, labelOrg = dataset
            predict = net(data.to(device))
            _, predicted = torch.max(predict.data, dim=1)
            totalAcc += predicted.cpu().eq(labelOrg).sum()
            label = F.one_hot(labelOrg.to(torch.long), 10).to(torch.float).to(device)
            sumLoss += lossFunc(predict, labelOrg).item()
            total += label.size(0)
    return sumLoss / len(testDataLoader), totalAcc / total

if __name__ == '__main__':
    # filePath = r"E:\dataset\SVHN\train_32x32.mat"
    save_path = "./net.pt"
    log_path = r"logs"
    require_improvement = 1000
    batchSize = 256
    n_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = models.resnet18()
    # 修改全连接层的输出
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)
    resnet18 = resnet18.to(device)
    # SVHNTrainData = SVHN(filePath)
    train_dataset = torchvision.datasets.SVHN(
        root='/home/star/Desktop/Res/dataset/SVHN',
        split='train',
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    test_dataset = torchvision.datasets.SVHN(
        root='/home/star/Desktop/Res/dataset/SVHN',
        split='test',
        download=False,
        transform=torchvision.transforms.ToTensor()
    )
    dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize, shuffle=True)
    optimizer = optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9)
    lossFunc = nn.CrossEntropyLoss()
    # train(resnet18, dataLoader, optimizer, lossFunc, n_epoch)


    #1.pt读取网络参数
    # model = resnet18
    # model_dict = torch.load("net.pt")
    # model.load_state_dict(model_dict)
    #2.pt读取整个网络
    # model = torch.load("resnet.pt")

    # model.eval()
    # data = np.load("test.npy")
    # data = data[None,:,:,:]
    # r = model(torch.tensor(data))
    # print(torch.argmax(r,1))

    #可视化输入图像
    # viz = visdom.Visdom(env="v")
    # viz.images(data,win="w")

    #onnx读取整个网络
    # torch.onnx.export(resnet18,(torch.rand(1,3,32,32)),"resNet.onnx",export_params=True,opset_version=10,input_names=["x1"],output_names=["out"])
    # model = onnx.load("resNet.onnx")
    # sim_model,_ = onnxsim.simplify(model)
    # onnx.save(sim_model,"resNet.onnx")
    # import onnxruntime
    # session = onnxruntime.InferenceSession("resNet.onnx")
    # input_name = session.get_inputs()[0].name
    # data = np.load("test.npy")
    # data = data[None,:,:,:]
    # out = session.run(None,{input_name:data})
    # out = torch.tensor(out)
    # out = torch.argmax(out[0])
    # print(out)