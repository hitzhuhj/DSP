import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from cifar_model import *
import argparse
import pickle
from sympy.abc import x
from sympy import Poly
import math
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

benchmark_mode(True)

parser = argparse.ArgumentParser(description='CIFAR-10 ResNet Training')
parser.add_argument('--save_dir', type=str, default='./cifarmodel/', help='Folder to save checkpoints and log.')
parser.add_argument('-l', '--layers', default=20, type=int, metavar='L', help='number of ResNet layers')
parser.add_argument('-d', '--device', default='0', type=str, metavar='D', help='main device (default: 0)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='J', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=164, type=int, metavar='E', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='B', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.device

h0 = 1
h1 = x
h2 = (x**2 - 1)/math.sqrt(math.factorial(2))
h3 = (x**3 - 3*x)/math.sqrt(math.factorial(3))
h4 = (x**4 - 6*x**2 + 3)/math.sqrt(math.factorial(4))
h5 = (x**5 - 10*x**3 +15*x)/math.sqrt(math.factorial(5))
h6 = (x**6 - 15*x**4 + 45*x**2 -15)/math.sqrt(math.factorial(6))
h7 = (x**7 - 21*x**5 + 105*x**3 - 105*x)/math.sqrt(math.factorial(7))
h8 = (x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105)/math.sqrt(math.factorial(8))
h9 = (x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x)/math.sqrt(math.factorial(9))
h10 = (x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 -945)/math.sqrt(math.factorial(10))
H_list = [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]

def load_variavle(filename):
    f = open(filename,"rb")
    r = pickle.load(f)
    f.close()
    return r

def hermite_act(order, hermite_params):
    temp = 0
    for i in range(order+1):
        temp += hermite_params[i]*H_list[i]
    return Poly(temp, x)

def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


def warmup(optimizer, lr, epoch):
    if epoch < 2:
        lr = lr/4
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if epoch == 2:
        lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

device = torch.device("cuda")

def train(filename, network):
    train_dataset = dsets.CIFAR10(root='/home/zhuhongjia/SSD/cifar10/',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                           std=(0.2470, 0.2435, 0.2616))
                                  ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, drop_last=True)
    test_dataset = dsets.CIFAR10(root='/home/zhuhongjia/SSD/cifar10/',
                               train=False,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2470, 0.2435, 0.2616))
                               ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False)


    cnn, netname = network
    config = netname
    for m in cnn.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)

    criterion = nn.CrossEntropyLoss()
    bestacc=0

    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, [args.epochs//2,args.epochs*3//4], 0.1)
    

    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        warmup(optimizer, args.lr, epoch)
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f" % (get_lr(optimizer)[0], loss.item(), bestacc))
            bar.update()

        scheduler.step()

        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()
        acc = 100 * correct / total
        cnn.train()

        if bestacc<acc:
            bestacc=acc
            torch.save([cnn.state_dict(),bestacc], args.save_dir+filename)
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f" % (get_lr(optimizer)[0], loss.item(), bestacc))
    bar.close()
    return bestacc

def resnet(layers):
    hermite_fit_params_path = '/home/zhuhongjia/SSD/D2B/relu/2orderHermite_relu_4.pkl'
    hermite_params = load_variavle(hermite_fit_params_path)
    return CifarResNet(ResNetBasicblock, layers, 10, hermite_params).to(device), "resnet"+str(layers)

if __name__ == '__main__':
    train('resnet%d.pkl'%(args.layers,), resnet(args.layers))