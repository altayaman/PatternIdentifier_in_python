from pybrain.datasets import SupervisedDataSet
from structures import *

dataModel = [
    [(0,0), (0,)],
    [(0,1), (1,)],
    [(1,0), (1,)],
    [(1,1), (0,)],
]

ds = SupervisedDataSet(2, 1)
for inpu, target in dataModel:
    ds.addSample(inpu, target)

# create a large random data set
import random
random.seed()
trainingSet = SupervisedDataSet(2, 1);
for ri in range(0,1000):
    inpu,target = dataModel[random.getrandbits(2)];
    trainingSet.addSample(inpu, target)

from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 2, 1, bias=True)

from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, learningrate = 0.001, momentum = 0.99)
trainer.trainUntilConvergence(verbose=True,
                              trainingData=trainingSet,
                              validationData=ds,
                              maxEpochs=10)

print('0,0->', net.activate([0,0]))
print('0,1->', net.activate([0,1]))
print('1,0->', net.activate([1,0]))
print('1,1->', net.activate([1,1]))