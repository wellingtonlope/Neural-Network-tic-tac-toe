def readFile(name):
    lines = open(name).readlines()
    for index, val in enumerate(lines):
        aux = val.split(',')
        for index2, val2 in enumerate(aux):
            aux[index2] = 1 if val2 == 'x' or val2 == 'positive\n' else -1 if val2 == 'o' or 'negative\n' else 0
        lines[index] = aux
    return lines

# x = 1 and o = -1 and b = 0
# positive = 1 and negative = 0
data = readFile('data/tic-tac-toe.data')
