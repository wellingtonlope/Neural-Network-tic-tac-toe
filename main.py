# x = 1 and o = -1 and b = 0
# positive = 1 and negative = 0
def readFile(name):
    lines = open(name).readlines()
    for index, val in enumerate(lines):
        aux = val.split(',')
        for index2, val2 in enumerate(aux):
            aux[index2] = 1 if val2 == 'x' or val2 == 'positive\n' else -1 if val2 == 'o' or 'negative\n' else 0
        lines[index] = aux
    return lines


from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from collections import Counter
from random import randint

dataTrain = readFile('data/tic-tac-toe.data')
dataTest = []

# separação dos dados de treino e de teste
length = len(dataTrain) - 1
for x in range(int(len(dataTrain) * 0.1)):
    rand = randint(0, length)
    dataTest.append(dataTrain.pop(rand))
    length -= 1

# criação do data set com 9 dados de entrada e 1 de saida
dsTrain = SupervisedDataSet(9, 1)
dsTest = SupervisedDataSet(9, 1)

# adicionando os dados ao dataset train
for x in dataTrain:
    dsTrain.appendLinked(x[:9], x[9:])

# adicionando os dados ao dataset test
for x in dataTest:
    dsTest.appendLinked(x[:9], x[9:])

# criação da rede
net = buildNetwork(9, 100, 1, bias=True)

# treinamento da rede neural
trainer = BackpropTrainer(net, dsTrain)
trainer.train()
trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=100, continueEpochs=10)

# testando rede
resTest = net.activateOnDataset(dsTest)

# verificando resultado
hit = []
for index, x in enumerate(resTest):
    hit.append(int(round(x[0], 0)) == dataTest[index][9])

result = Counter(hit)
print('True = {:.2f}'.format(result[True] / len(hit)))
print('False = {:.2f}'.format(result[False] / len(hit)))
