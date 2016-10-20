import numpy as np
import copy as cpy
import matplotlib.pyplot as plt

def MatrixCreate(rows, columns):
    return np.zeros(shape=(rows,columns))

def VectorCreate(width):
    return np.zeros(width, dtype='f')

def MatrixRandomize(matrix):
    newmat = cpy.deepcopy(matrix)
    for i in np.nditer(newmat, op_flags=['readwrite']):
        # Random number between [-1, 1)
        i[...] = -2 * np.random.random_sample() + 1
    return newmat

def MatrixPerturb(p, prob):
    modifiedMatrix = cpy.deepcopy(p)
    for i in np.nditer(modifiedMatrix, op_flags=['readwrite']):
        # introduce error in copy, if probability is > rand.
        # allow for a new random number.
        if prob > np.random.random():
            i[...] = np.random.random()
    return modifiedMatrix

def NeuralUpdate(neuronValues, synapses, i):
    # Computer New values of all neurons
    for j in xrange(numNeurons):
        temp = 0
        for k in xrange(numNeurons):
            temp += neuronValues[i-1][k]*synapses[j][k]
        # Squish function
        if temp < 0:
            temp = 0
        elif temp > 1:
            temp = 1
        neuronValues[i][j] = temp

def MeanDistance(v1, v2):
    d = 0
    for i in xrange(numNeurons):
        d += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return d/numNeurons

def AvgDistNeighbors(neuronValues):
    diff = 0.0
    for i in range(1, 9):
        for j in range(0, 9):
            diff = diff + abs(neuronValues[i][j] - neuronValues[i][j+1])
            diff = diff + abs(neuronValues[i+1][j] - neuronValues[i][j])
    diff = diff/(2*8*9)
    return diff

def Fitness(v, img=0):
    neuronValues = MatrixCreate(numNeurons, numNeurons)
    for i in range(numNeurons):
        neuronValues[0][i] = .5
    for i in range(1, numNeurons):
        NeuralUpdate(neuronValues, v, i)

    # After updates, if image flag passed, present visualized data
    if img == 1:
        plt.imshow(neuronValues, cmap='gray', aspect='auto', interpolation='nearest')
        plt.show()
    # Grab the last row vector of neuronValues, The ending values after all updates.
    actualNeuronValues = neuronValues[9,:]
    # Compare to desired neuronValues
    # return 1 - distance, to imply fitness since mean squared will return 0 -> 1
    return 1 - AvgDistNeighbors(neuronValues)

def Fitness2(v, img=0):
    neuronValues = MatrixCreate(numNeurons, numNeurons)
    for i in range(numNeurons):
        neuronValues[0][i] = .5
    for i in range(1, numNeurons):
        NeuralUpdate(neuronValues, v, i)

    # After updates, if image flag passed, present visualized data
    if img == 1:
        plt.imshow(neuronValues, cmap='gray', aspect='auto', interpolation='nearest')
        plt.show()
    # Grab the last row vector of neuronValues, The ending values after all updates.
    actualNeuronValues = neuronValues[9,:]
    # Compare to desired neuronValues
    # return 1 - distance, to imply fitness since mean squared will return 0 -> 1
    return 1 - MeanDistance(actualNeuronValues, DesiredNeuronValues)

if __name__ == '__main__':
    fitnessVector = []
    DesiredNeuronValues = VectorCreate(10);
    for j in range(1,10,2):
        DesiredNeuronValues[j] = 1
    numNeurons = 10
    parent = MatrixCreate(numNeurons, numNeurons) #Synapses
    parent = MatrixRandomize(parent)
    # Confirm Parent was properly randomized
    # print parent
    parentFitness = Fitness2(parent, 1)

    # print parentFitness
    for currentGeneration in range(0,100000):
        print currentGeneration, parentFitness
        fitnessVector.append(parentFitness)
        child = MatrixPerturb(parent, 0.05)
        childFitness = Fitness2(child)
        if (childFitness > parentFitness):
            parent = child
            parentFitness = childFitness
    Fitness(parent, 1)
