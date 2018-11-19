from algorithm import readFile, separator, trainer
#leitura e conversão dos dados
data = readFile('data/tic-tac-toe.data')

#separação dos dados de treinamento e teste
dataTrain, dataTest = separator(data, 320)

# treinamento e obtenção dos resultados da rede neural
# trainer(dadosDeTreinamento, dadosDeTeste, neuroniosNaCamadaOculta, taxaDeAprendizado, geracoes)
result = trainer(dataTrain, dataTest, 75, 0.4, 100)
print(result.treino)
print(result.teste)
