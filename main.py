from algorithm import readFile, separator, trainer
data = readFile('data/tic-tac-toe.data')

dataTrain, dataTest = separator(data, 320)

result = trainer(dataTrain, dataTest, 75, 0.4, 100)
print(result.treino)
print(result.teste)
