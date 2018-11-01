import torch
import numpy as np

device = torch.device('cpu')

EncoderIn, EncoderH, EncoderOut = 1, 9, 9
EncoderWeight1 = torch.randn(EncoderIn, EncoderH, device = device, requires_grad = True)
EncoderWeight2 = torch.randn(EncoderH, EncoderOut, device = device, requires_grad = True)

DecoderIn, DecoderH, DecoderOut = 9, 9, 1
DecoderWeight1 = torch.randn(DecoderIn, DecoderH, device = device, requires_grad = True)
DecoderWeight2 = torch.randn(DecoderH, DecoderOut, device = device, requires_grad = True)

u = torch.randn(200, 3, device = device, requires_grad = True)
v = torch.randn(200, 3, device = device, requires_grad = True)
w = torch.randn(200, 3, device = device, requires_grad = True)

LearningRate = 0.01

def files(filename):
    BigData = filename.read().split()
    Data = []
    X = []
    for i in range(len(BigData)):
        Line = BigData[i].split(",")
        Data.append(Line[3])
        X.append(Line[0])
        X.append(Line[1])
        X.append(Line[2])
    Inputs = np.array(Data, dtype = np.float64)
    shape = len(Inputs), 1
    Inputs = Inputs.reshape(shape)
    X = np.array(X, dtype = np.float64)
    X = X.reshape(len(Inputs), 3)
    return (Inputs, X)

def TensorData(filename):
    BigData = filename.read().split()
    TData = []
    for i in range(len(BigData)):
        Line = BigData[i].split(",")
        TData.append(Line[0])
        TData.append(Line[1])
        TData.append(Line[2])
    TData = np.array(TData, dtype = np.float64)
    TData = TData.reshape(len(BigData), 3)
    return (TData)

traindata = open("train-fold-1.txt", "r")
Inputs, X = files(traindata)
Inputs = torch.from_numpy(Inputs).float()

UFile = open("U-init-fold-1-mode-1.txt", "r")
UData = TensorData(UFile)
UData = torch.from_numpy(UData).float()

for t in range(20):
    EncoderOutput = Inputs.mm(EncoderWeight1).clamp(min = 0).mm(EncoderWeight2)
    DecoderOutput = EncoderOutput.mm(DecoderWeight1).clamp(min = 0).mm(DecoderWeight2)
    loss = (DecoderOutput - Inputs).pow(2).mean()
    print(t, loss.item())
    loss.backward()
    #UCal = EncoderOutput[:, :3]
    #ULoss = (UData - UCal).pow(2).mean()
    #print ULoss
    with torch.no_grad():
        EncoderWeight1 -= LearningRate * EncoderWeight1.grad
        EncoderWeight2 -= LearningRate * EncoderWeight2.grad
        DecoderWeight1 -= LearningRate * DecoderWeight1.grad
        DecoderWeight2 -= LearningRate * DecoderWeight2.grad
        #u -= LearningRate * u.grad
        #v -= LearningRate * v.grad
        #w -= LearningRate * w.grad
        EncoderWeight1.grad.zero_()
        EncoderWeight2.grad.zero_()
        DecoderWeight1.grad.zero_()
        DecoderWeight2.grad.zero_()

#testdata = open("test-fold-1.txt", "r")
#Inputs, X = files(testdata)
#Inputs = torch.from_numpy(Inputs).float()
