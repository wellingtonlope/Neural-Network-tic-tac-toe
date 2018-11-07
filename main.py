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


def separateInOut(data):
    i = []
    o = []
    for x in data:
        i.append(x[:9])
        o.append(x[9:])

    return i, o


from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from collections import Counter

data = readFile('data/tic-tac-toe.data')

# criação do data set com 9 dados de entrada e 1 de saida
ds = SupervisedDataSet(9, 1)

# adicionando os dados ao dataset
for x in data:
    ds.appendLinked(x[:9], x[9:])

# criação da rede
net = buildNetwork(9, 100, 1, bias=True)

# treinamento da rede neural
trainer = BackpropTrainer(net, ds)
trainer.train()
trainer.trainUntilConvergence(verbose=True, validationProportion=0.15, maxEpochs=10, continueEpochs=10)

# testando rede
p = net.activateOnDataset(ds)

# verificando resultado
v = []
for index, x in enumerate(p):
    v.append(int(round(x[0], 0)) == data[index][9])

result = Counter(v)
print('True = {:.2f}'.format(result[True] / len(v)))
print('False = {:.2f}'.format(result[False] / len(v)))
