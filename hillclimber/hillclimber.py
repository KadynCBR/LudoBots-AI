import numpy as np
import copy as cpy
import matplotlib.pyplot as plt

def MatrixCreate(rows, columns):
    return np.zeros(shape=(rows,columns))

def MatrixRandomize(matrix):
    newmat = cpy.deepcopy(matrix)
    for i in np.nditer(newmat, op_flags=['readwrite']):
        i[...] = np.random.random()
    return newmat

def fitness(v):
    summation = 0
    count = 0
    for i in np.nditer(v, op_flags=['readwrite']):
        summation += i[...]
        count += 1
    return summation/count

def MatrixPerturb(p, prob):
    modifiedMatrix = cpy.deepcopy(p)
    for i in np.nditer(modifiedMatrix, op_flags=['readwrite']):
        # introduce error in copy, if probability is > rand.
        # allow for a new random number.
        if prob > np.random.random():
            i[...] = np.random.random()
    return modifiedMatrix

def PlotVectorAsLine(fitnesses):
    for fits in fitnesses:
        plt.plot(fits[0])
        plt.ylabel('Fitness')
        plt.xlabel('Generation')
    plt.show()

Genes = MatrixCreate(50, 5000)
fitnesses = []
# for i in range (5):
parent = MatrixCreate(1, 50)
parent = MatrixRandomize(parent)
parentFitness = fitness(parent)
fits = MatrixCreate(1, 5000);

for currentGeneration in range(5000):
    print currentGeneration, parentFitness
    fits[0][currentGeneration] = parentFitness
    child = MatrixPerturb(parent,0.05)
    childFitness = fitness(child)
    if childFitness > parentFitness:
        parent = child
        parentFitness = childFitness
    for j in range(50):
        Genes[j][currentGeneration] = parent[0][j]
fitnesses.append(fits);
print Genes

# PlotVectorAsLine(fitnesses)
plt.imshow(Genes, cmap='gray', aspect='auto', interpolation='nearest')
plt.show()
