import tensorflow as tf
import numpy as np

Inputs = tf.placeholder(tf.float64)
a = tf.placeholder(tf.float64)
b = tf.placeholder(tf.float64)
c = tf.placeholder(tf.float64)
d = tf.placeholder(tf.float64)

sampledata = open("traindata.txt", "r")

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

u = np.random.rand(200,3)
v = np.random.rand(200,3)
w = np.random.rand(200,3)

seed = 128
epochs = 10
LearningRate = tf.Variable(0.01, dtype = tf.float64)

xEncoder = tf.placeholder(tf.float32)
InputUnitsEncoder = 1
HiddenUnitsEncoder = 9
OutputUnitsEncoder = 9

WeightsEncoder = {
    'hidden': tf.Variable(tf.random_normal([InputUnitsEncoder, HiddenUnitsEncoder])),
    'output': tf.Variable(tf.random_normal([HiddenUnitsEncoder, OutputUnitsEncoder]))
}

BiasesEncoder = {
    'hidden': tf.Variable(tf.random_normal([HiddenUnitsEncoder])),
    'output': tf.Variable(tf.random_normal([OutputUnitsEncoder]))
}

HiddenLayerEncoder = tf.add(tf.matmul(xEncoder, WeightsEncoder['hidden']), BiasesEncoder['hidden'])
HiddenLayerEncoder = tf.nn.relu(HiddenLayerEncoder)
OutputLayerEncoder = tf.matmul(HiddenLayerEncoder, WeightsEncoder['output']) + BiasesEncoder['output']
OutputLayerEncoder = tf.nn.relu(OutputLayerEncoder)

xDecoder = tf.placeholder(tf.float32)
InputUnitsDecoder = 9
HiddenUnitsDecoder = 3
OutputUnitsDecoder = 1

WeightsDecoder = {
    'hidden': tf.Variable(tf.random_normal([InputUnitsDecoder, HiddenUnitsDecoder])),
    'output': tf.Variable(tf.random_normal([HiddenUnitsDecoder, OutputUnitsDecoder]))
}

BiasesDecoder = {
    'hidden': tf.Variable(tf.random_normal([HiddenUnitsDecoder])),
    'output': tf.Variable(tf.random_normal([OutputUnitsDecoder]))
}

HiddenLayerDecoder = tf.add(tf.matmul(xDecoder, WeightsDecoder['hidden']), BiasesDecoder['hidden'])
HiddenLayerDecoder = tf.nn.relu(HiddenLayerDecoder)
OutputLayerDecoder = tf.matmul(HiddenLayerDecoder, WeightsDecoder['output']) + BiasesDecoder['output']
OutputLayerDecoder = tf.nn.relu(OutputLayerDecoder)

LossCal = tf.reduce_mean(tf.square(a - b))
GradCal = tf.gradients(LossCal, a)
WeightUpdate1 = tf.assign(WeightsEncoder['hidden'], tf.subtract(WeightsEncoder['hidden'],tf.multiply(LearningRate, c)))
WeightUpdate2 = tf.assign(WeightsEncoder['output'], tf.subtract(WeightsEncoder['output'],tf.multiply(LearningRate, c)))

XGrad = tf.gradients(LossCal, d)
UUpdate = tf.assign(u, tf.subtract(u, tf.multiply(LearningRate, c)))
VUpdate = tf.assign(v, tf.subtract(v, tf.multiply(LearningRate, c)))
WUpdate = tf.assign(w, tf.subtract(w, tf.multiply(LearningRate, c)))
WeightUpdate3 = tf.assign(WeightsDecoder['hidden'], tf.subtract(WeightsDecoder['hidden'],tf.multiply(LearningRate, c)))
WeightUpdate4 = tf.assign(WeightsDecoder['output'], tf.subtract(WeightsDecoder['output'],tf.multiply(LearningRate, c)))

#Assignment = tf.subtract(a, tf.multiply(LearningRate, c))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

Inputs, X = files(sampledata)

##Training Phase
for k in range(epochs):
    #ShapedInput = np.array(Inputs[j], dtype = np.float64)
    #ShapedInput.shape = (1,1)
    val1 = sess.run(OutputLayerEncoder, feed_dict = {xEncoder: Inputs})
    val2 = sess.run(OutputLayerDecoder, feed_dict = {xDecoder: val1})
    loss1, grad1 = sess.run([LossCal, GradCal], feed_dict = {a: Inputs, b: val2})
    sess.run([WeightUpdate1, WeightUpdate2], feed_dict = {c: grad1})
    loss2, grad2 = sess.run([LossCal, XGrad], feed_dict = {a: Inputs, b: val2, d: X})
    sess.run([UUpdate, VUpdate, WUpdate, WeightUpdate3, WeightUpdate4], feed_dict = {c: grad2})
    #print grad
    #sess.run([WeightUpdate1, WeightUpdate2], feed_dict = {c: grad})
    #val3 = sess.run(Assignment, feed_dict = {a: Inputs, c: grad})
    #Inputs[j] = val3
    '''XConcat = []
    uval = X[j,0]
    vval = X[j,1]
    wval = X[j,2]
    uval = np.int_(uval)
    vval = np.int_(vval)
    wval = np.int_(wval)
    user = u[uval,]
    XConcat.append(user[0])
    XConcat.append(user[1])
    XConcat.append(user[2])
    item = v[vval,]
    location = w[wval,]
    print XConcat
    loss1, grad1 = sess.run([L, gu], feed_dict = {a: ShapedInput, b:val2, d: XConcat})'''
    #loss1, grad1 = sess.run([LossCal, gu], feed_dict = {a:ShapedInput, b: val2, d: XConcat})

    #grad1 = sess.run(gu, feed_dict = {LossCal: loss})
    #loss, grad1 = sess.run([LossCal, gu], feed_dict = {a:Inputs, b: val2})
    #loss, grad2 = sess.run([LossCal, gv], feed_dict = {a:Inputs, b: val2})
    #loss, grad3 = sess.run([LossCal, gw], feed_dict = {a:Inputs, b: val2})
#_, c = sess.run([optimizer, loss])
'''grad1 = tf.assign(WeightsEncoder['hidden'], tf.subtract(WeightsEncoder['hidden'],tf.multiply(LearningRate, c)))
grad2 = tf.assign(WeightsEncoder['output'], tf.subtract(WeightsEncoder['output'],tf.multiply(LearningRate, c)))
sess.run(grad1)
sess.run(grad2)'''

##Testing Phase
'''TestVal = files(TestFile1)
val1 = sess.run(OutputLayerEncoder, feed_dict = {xEncoder: TestVal})
val2 = sess.run(OutputLayerDecoder, feed_dict = {xDecoder: val1})
loss = tf.reduce_mean(tf.square(TestVal - val2))
res = sess.run(loss)'''
#print res
'''if i == 1:
    val = files(TrainFile1)
if i == 2:
    val = files(TrainFile2)
if i == 3:
    val = files(TrainFile3)
if i == 4:
    val = files(TrainFile4)
if i == 5:
    val = files(TrainFile5)
if i == 6:
    break
if i == 1:
    TestVal = files(TestFile1)
if i == 2:
    TestVal = files(TestFile2)
if i == 3:
    TestVal = files(TestFile3)
if i == 4:
    TestVal = files(TestFile4)
if i == 5:
    TestVal = files(TestFile5)
if i == 6:
    break

WeightEncoderHidden = sess.run(WeightsEncoder['hidden'])
WeightEncoderOutput = sess.run(WeightsEncoder['output'])
NewVal1 = tf.cast(val1, dtype = tf.int32)
NewVal1 = tf.abs(NewVal1)
tensor = sess.run(NewVal1)
uval = tensor[0,0]
u[uval,0] = ShapedInput * WeightEncoderHidden[0,0] * WeightEncoderOutput[0,0]
u[uval,1] = ShapedInput * WeightEncoderHidden[0,1] * WeightEncoderOutput[0,1]
u[uval,2] = ShapedInput * WeightEncoderHidden[0,2] * WeightEncoderOutput[0,2]
vval = tensor[0,1]
v[vval,0] = ShapedInput * WeightEncoderHidden[0,0] * WeightEncoderOutput[1,0]
v[vval,1] = ShapedInput * WeightEncoderHidden[0,1] * WeightEncoderOutput[1,1]
v[vval,2] = ShapedInput * WeightEncoderHidden[0,2] * WeightEncoderOutput[1,2]
wval = tensor[0,2]
w[wval,0] = ShapedInput * WeightEncoderHidden[0,0] * WeightEncoderOutput[2,0]
w[wval,1] = ShapedInput * WeightEncoderHidden[0,1] * WeightEncoderOutput[2,1]
w[wval,2] = ShapedInput * WeightEncoderHidden[0,2] * WeightEncoderOutput[2,2]

TrainFile1 = open("train-fold-1.txt", "r")
TrainFile2 = open("train-fold-2.txt", "r")
TrainFile3 = open("train-fold-3.txt", "r")
TrainFile4 = open("train-fold-4.txt", "r")
TrainFile5 = open("train-fold-5.txt", "r")
TestFile1 = open("test-fold-1.txt", "r")
TestFile2 = open("test-fold-1.txt", "r")
TestFile3 = open("test-fold-1.txt", "r")
TestFile4 = open("test-fold-1.txt", "r")
TestFile5 = open("test-fold-1.txt", "r")

def variance(x):
    n = len(x)
    x_bar = sum(x)/n
    return round(sum((x_i - x_bar)**2 for x_i in x)/(n-1), 2)'''
