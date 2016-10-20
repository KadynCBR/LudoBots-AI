import numpy as np
import copy as cpy
import math as math
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

def PlotNeuronPositions(neuronPos, numNeu, synapses):
    for i in range(0, numNeu):
        x = neuronPositions[0][i]
        y = neuronPositions[1][i]
        plt.plot(x,y, 'ko', markerfacecolor=[1,1,1], markersize=18)
    for i in range(0, numNeu):
        for j in range(0, numNeu):
            w = int(10*abs(synapses[i][j]))+1
            if (synapses[i][j] > 0):
                plt.plot([neuronPos[0][i],neuronPos[0][j]], [neuronPos[1][i],neuronPos[1][j]], color=[0.8, 0.8, 0.8], linewidth = w)
            else:
                plt.plot([neuronPos[0][i],neuronPos[0][j]], [neuronPos[1][i],neuronPos[1][j]], color=[0, 0, 0], linewidth = w)
    plt.show()

def NeuralUpdate(neuronValues, synapses, i):
    # Computer New values of all neurons
    for j in range(9):
        temp = 0
        for k in range(9):
            temp += neuronValues[i-1][k]*synapses[j][k]
        # Squish function
        if temp < 0:
            temp = 0
        elif temp > 1:
            temp = 1
        neuronValues[i][j] = temp
##################
# Initialize Neurons
numNeurons = 10
neuronValues = MatrixCreate(50,numNeurons)
# Set initial status to randoms
for i in range(numNeurons):
    neuronValues[0][i] = np.random.random()

#matrix for neuron Network visualizations
neuronPositions = MatrixCreate(2, numNeurons)
angle = 0.0
angleUpdate = 2 * math.pi/numNeurons
for i in range(0, numNeurons):
    x = math.sin(angle)
    y = math.cos(angle)
    angle = angle + angleUpdate
    neuronPositions[0][i] = x
    neuronPositions[1][i] = y

synapses = MatrixCreate(10,10)
for i in np.nditer(synapses, op_flags=['readwrite']):
    i[...] = -2 * np.random.random_sample() + 1

for i in xrange(1,49):
    NeuralUpdate(neuronValues,synapses,i)

# Working
# PlotNeuronPositions(neuronPositions, numNeurons, synapses)
plt.imshow(neuronValues, cmap='gray', aspect='auto', interpolation='nearest')
plt.show()
