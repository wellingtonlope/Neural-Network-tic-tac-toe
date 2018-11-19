from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from collections import Counter
from random import randint

class Resultado:
    def __init__(self, treino, teste):
        self.treino = treino
        self.teste = teste

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

# separação dos dados de treino e de teste
def separator(data, lengthTrain):
    dataTest = []
    length = len(data) - 1
    for x in range(len(data) - lengthTrain):
        rand = randint(0, length)
        dataTest.append(data.pop(rand))
        length -= 1
    return data, dataTest

def trainer(dataTrain, dataTest, hiddenLayer, validationProportion, epochs):
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
    net = buildNetwork(9, hiddenLayer, 1, bias=True)

    # treinamento da rede neural
    trainer = BackpropTrainer(net, dsTrain)
    trainer.train()
    trainer.trainUntilConvergence(verbose=False, validationProportion=validationProportion, maxEpochs=epochs)

    # testando rede
    resTrain = net.activateOnDataset(dsTrain)
    resTest = net.activateOnDataset(dsTest)

    # verificando resultado
    hitTrain = []
    for index, x in enumerate(resTrain):
        hitTrain.append(int(round(x[0], 0)) == dataTrain[index][9])

    hitTest = []
    for index, x in enumerate(resTest):
        hitTest.append(int(round(x[0], 0)) == dataTest[index][9])

    resultTrain = Counter(hitTrain)
    resultTest = Counter(hitTest)

    return Resultado({'acertos': resultTrain[True] / len(hitTrain), 'erros': resultTrain[False] / len(hitTrain)}, {'acertos': resultTest[True] / len(hitTest), 'erros': resultTest[False] / len(hitTest)})
