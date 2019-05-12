# Kevin Sittser
# 525003900
# CSCE 420
# Due: April 25, 2019
# network.py


import numpy
import random
import bdfparse
import sys

tanhTable = []
sigmoidTable = []
sigmoidTableNeg = []

def sigmoid(x):
    '''Perform the sigmoid function, creating tabulated data to use for future calls'''
    if x >= 0:
        if len(sigmoidTable) < int(abs(round(x*1000)))+1:
            for i in range(len(sigmoidTable),int(abs(round(x*1000)))+1):
                sigmoidTable.append(1/(1+numpy.exp(-i/1000)))
    else:
        if len(sigmoidTableNeg) < int(abs(round(x*1000)))+1:
            for i in range(len(sigmoidTableNeg),int(abs(round(x*1000)))+1):
                sigmoidTableNeg.append(1/(1+numpy.exp(i/1000)))
    if x >= 0:
        return sigmoidTable[int(abs(round(x*1000)))]
    else:
        return sigmoidTableNeg[int(abs(round(x*1000)))]
    
def derivSigmoid(x):
    '''Perform the derivative of the sigmoid function'''
    return sigmoid(x)*(1-sigmoid(x))
    
def tanh(x):
    '''Perform the hyperbolic tangent function, creating tabulated data to use for future calls'''
    if len(tanhTable) < int(abs(round(x*1000)))+1:
        for i in range(len(tanhTable),int(abs(round(x*1000)))+1):
            tanhTable.append(numpy.tanh(i/1000))
    if x >= 0:
        return tanhTable[int(abs(round(x*1000)))]
    else:
        return -tanhTable[int(abs(round(x*1000)))]
    
def derivTanh(x):
    '''Perform the derivative of the tanh function'''
    return 1-(tanh(x))**2
    
def letterToList(letter):
    '''Return a list of 26 numbers, all of which are zero except the one corresponding to input letter, which is one'''
    l = [0] * 26
    l[ord(letter)-ord('A')] = 1
    return l

class Neuron:
    '''An individual neuron, containing activation value and weights connecting it to each neuron in the next layer'''
    def __init__(self):
        self.activation = None
        self.weights = []  # weights for connecting to *next* layer
    
class Layer:
    '''A layer of the neural network, containing a list of neurons and the activation function used for the neurons in the next layer'''
    def __init__(self,size,nextSize,actFunction,actDeriv):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron())
        for n in self.neurons:
            n.weights = []
            for _ in range(nextSize):
                n.weights.append(None)
        self.size = size
        self.actFunction = actFunction  # activation function for activating *next* layer
        self.actDeriv = actDeriv
            
class Network:
    '''A neural network, containing several layers of neurons, as well as a set of example data'''
    def __init__(self,layerData,examples):
        self.layers = []
        for layer in layerData:
            self.layers.append(Layer(layer[0],layer[1],layer[2],layer[3]))
        self.examples = examples
        self.learningRate = 0.2
        
    def testInput(self,input):
        ''' Once the network is trained, find its output for some data'''
        
        # FEED FORWARD
        # transfer letter bitmap to input layer's activations
        activations = input
        for i in range(len(activations)):
            self.layers[0].neurons[i].activation = activations[i]
        # set the always-1 neuron's activation
            self.layers[0].neurons[-1].activation = 1
        # calculate activations for other layers
        for lr in range(1,len(self.layers)):
            for nr in range(len(self.layers[lr].neurons)):
                sum = 0  # extra always-one input
                for pnr in range(len(self.layers[lr-1].neurons)):
                    sum += self.layers[lr-1].neurons[pnr].weights[nr]*self.layers[lr-1].neurons[pnr].activation
                self.layers[lr].neurons[nr].activation = self.layers[lr-1].actFunction(sum)
            # set the always-1 neurons' activations
            if lr < len(self.layers)-1:
                self.layers[lr].neurons[-1].activation = 1
        return [neuron.activation for neuron in self.layers[-1].neurons]
    def train(self,epochs):
        '''Train network for input number of epochs based on given examples with answers'''
        
        # initialize with small random weights
        for lr in range(len(self.layers)):
            for nr in range(len(self.layers[lr].neurons)):
                for wt in range(len(self.layers[lr].neurons[nr].weights)):
                    self.layers[lr].neurons[nr].weights[wt] = random.uniform(-0.001,0.001)
        print('Training for',epochs,'epochs:')
        for i in range(epochs):
            if not ((100*(i+1)/epochs) % 10):
                print(str(i+1)+'/'+str(epochs)+' epochs done')
            inputs = []
            for lr in range(len(self.layers)):
                inputs.append([])
                if lr > 0:
                    for _ in self.layers[lr].neurons:
                        inputs[-1].append(None)
            for letter,bitmap in self.examples:
                
                # FEED FORWARD
                # transfer letter bitmap to input layer's activations
                activations = bitmap
                for ac in range(len(activations)):
                    self.layers[0].neurons[ac].activation = activations[ac]
                # set the always-1 neuron's activation
                self.layers[0].neurons[-1].activation = 1
                # calculate activations for other layers
                for lr in range(1,len(self.layers)):
                    for nr in range(len(self.layers[lr].neurons)):
                        sum = 0  # extra always-one input
                        for pnr in range(len(self.layers[lr-1].neurons)):
                            sum += self.layers[lr-1].neurons[pnr].weights[nr]*self.layers[lr-1].neurons[pnr].activation
                        inputs[lr][nr] = sum
                        self.layers[lr].neurons[nr].activation = self.layers[lr-1].actFunction(sum)
                    # set the always-1 neurons' activations
                    if lr < len(self.layers)-1:
                        self.layers[lr].neurons[-1].activation = 1
                
                # BACK-PROPAGATE
                Deltas = []
                for layer in self.layers:
                    Deltas.append([])
                    for _ in layer.neurons:
                        Deltas[-1].append(None)
                # calculate Deltas for output layer
                for nr in range(len(self.layers[-1].neurons)):
                    Deltas[-1][nr] = self.layers[-2].actDeriv(inputs[-1][nr]) * (letter[nr] - self.layers[-1].neurons[nr].activation)
                # calculate Deltas for other layers
                for lr in range(len(self.layers)-2,0,-1):
                    for nr in range(len(self.layers[lr].neurons)):
                        sum = 0
                        for wt in range(len(self.layers[lr].neurons[nr].weights)):
                            sum += self.layers[lr].neurons[nr].weights[wt] * Deltas[lr+1][wt]
                        Deltas[lr][nr] = self.layers[lr-1].actDeriv(inputs[lr][nr]) * sum
                for lr in range(len(self.layers)-1):
                    for nr in range(len(self.layers[lr].neurons)):
                        for wt in range(len(self.layers[lr].neurons[nr].weights)):
                            self.layers[lr].neurons[nr].weights[wt] += self.learningRate * self.layers[lr].neurons[nr].activation * Deltas[lr+1][wt]
    def printAllWeights(self):
        '''Print out all weights in all neurons of network, for testing purposes'''
        for i in range(len(self.layers)):
            print('LAYER',i)
            for neuron in self.layers[i].neurons:
                print('  neuron',neuron.activation,[weight for weight in neuron.weights])
    def writeToFile(self,fname,epochs):
        '''Write network data to a file'''
        f = open(fname,'w')
        f.write('-- Trained for '+str(epochs)+' epochs with learning rate alpha = '+str(self.learningRate)+' --\n\n')
        f.write('SIZES: ')
        f.write(' '.join([str(layer.size) for layer in self.layers]))
        f.write('\n')
        for lr in range(len(self.layers)):
            f.write('LAYER '+str(lr)+'\n')
            for neuron in self.layers[lr].neurons:
                f.write(' '.join([str(weight) for weight in neuron.weights])+'\n')
        for example in self.examples:
            f.write('EXAMPLE\n')
            f.write(' '.join([str(n) for n in example[0]])+' ; ')
            f.write(' '.join([str(n) for n in example[1]])+'\n')
    def initializeFromFile(self,fname):
        '''Initialize the network from file data'''
        f = open(fname,'r')
        lines = f.read().split('\n')[2:-1]
        lr = 0
        nr = 0
        sizes = lines[0].split()[1:]
        sizes = [int(size) for size in sizes]
        self.examples = []
        self.layers = [Layer(sizes[i],sizes[i+1],tanh,derivTanh) for i in range(len(sizes)-2)]
        self.layers.append(Layer(sizes[-2],sizes[-1],sigmoid,derivSigmoid))
        self.layers.append(Layer(sizes[-1],0,None,None))
        examplesStart = False
        for line in lines[1:]:
            if line[:5] == 'LAYER':
                lr = int(line[6:])
                nr = 0
            elif line[:7] == 'EXAMPLE':
                examplesStart = True
            elif not examplesStart:
                wts = line.split()
                for wt in range(len(wts)):
                    self.layers[lr].neurons[nr].weights[wt] = float(wts[wt])
                nr += 1
            else:
                ex = [part.split() for part in line.split(' ; ')]
                self.examples.append(([int(a) for a in ex[0]],[float(a) for a in ex[1]]))
    def randomTests(self,num,bitFlips):
        '''Test lots of random letters in the network, flipping some random bits, and return how many it got right'''
        correct = 0
        print('Testing accuracy for',bitFlips,'dot flips:')
        for i in range(num):
            if not ((100*(i+1)/num) % 10):
                print(str(i+1)+'/'+str(num)+' tests done')
            idx = random.randint(0,len(self.examples)-1)
            letter = self.examples[idx][1][:]
            for _ in range(bitFlips):
                flip = random.randint(0,len(self.examples[idx][1])-1)
                letter[flip] = float(not letter[flip])
            arr = self.testInput(letter)
            # for i in range(len(letter)//9):
                # print(''.join([('#' if letter[9*i+k] else ' ') for k in range(9)]),'      ',''.join([('#' if self.examples[idx][1][9*i+k] else ' ') for k in range(9)]))
            # print('===============',idx+1,numpy.argmax(arr)+1)
            if numpy.argmax(arr) == idx:
                correct += 1
        return correct
    def testLetter(self,letterIdx):
        '''Find average number of flips a letter requires for answer to not be consistently correct'''
        bitFlips = 0
        while True:
            correct = 0
            for _ in range(50):
                # test 50 different arrangements of bit flips
                letter = self.examples[letterIdx][1][:]
                for _ in range(bitFlips):
                    flip = random.randint(0,len(self.examples[letterIdx][1])-1)
                    letter[flip] = float(not letter[flip])
                arr = self.testInput(letter)
                if numpy.argmax(arr) == letterIdx:
                    correct += 1
            if correct < 48:
                # fail if fewer than 95% of attempts were identified correctly
                break
            bitFlips += 1
        return bitFlips

def network(inData,layerSizes,epochs):
    '''Create a network with given example data and layer sizes'''
    l = []
    # add extra neuron that will have activation always 1
    for i in range(len(layerSizes)-1):
        layerSizes[i] += 1
    for i in range(len(layerSizes)-2):
        l.append((layerSizes[i],layerSizes[i+1],tanh,derivTanh))
    l.append((layerSizes[-2],layerSizes[-1],sigmoid,derivSigmoid))
    l.append((layerSizes[-1],0,None,None))
    network = Network(l,inData)
    network.train(epochs)
    return network
    
def letterNetwork(epochs):
    '''Create a network for reading letters of the alphabet, and train for input number of epochs'''
    inData = bdfparse.letterMap()
    inData = [(letterToList(letter),bitmap) for letter,bitmap in inData.items()]
    inData = [(letter,[float(neurAct) for row in bitmap for neurAct in row]) for letter,bitmap in inData]
    layerSizes = [126,126,26]  # where first layer is input layer, last layer is output layer
    net = network(inData,layerSizes,epochs)
    return net
    
def majorityNetwork(epochs):
    '''Create a network for the majority function, and train for input number of epochs'''
    inData = [([1,0],[0,0,0]),([1,0],[0,0,1]),([1,0],[0,1,0]),([0,1],[0,1,1]),([1,0],[1,0,0]),([0,1],[1,0,1]),([0,1],[1,1,0]),([0,1],[1,1,1])]
    layerSizes = [3,1,2]
    net = network(inData,layerSizes,epochs)
    return net
    
if __name__ == '__main__':
    '''If a file name is specified in command line, load from that file; otherwise create a new neural network and train it.  Then test the neural network for its accuracy with input number of dot flips'''
    net = Network([],[])
    if len(sys.argv) > 1:
        # load neural network from specified file
        print('Initializing neural network from file:',sys.argv[1])
        net.initializeFromFile(sys.argv[1])
    else:
        # make a new neural netwrk
        print('Creating new neural network')
        epochs = 300
        net = letterNetwork(epochs)
        fname = 'data1.txt'
        print('Writing to file:',fname)
        net.writeToFile(fname,epochs)
    flips = int(input('How many dot flips?: '))
    k = 1000
    res = net.randomTests(k,flips)
    print('\nCorrect: ',str(res)+'/'+str(k)+' = '+str(100*res//k)+' % accuracy')
        
    # net = Network([],[])
    # net.initializeFromFile()
    # print('res',net.testInput(net.examples[0][1]))
    
    # net = letterNetwork()
    # net.writeToFile()
    
    # net = Network([],[])
    # net.initializeFromFile()
    # net.printAllWeights()
    # print('res',numpy.argmax(net.testInput(net.examples[20][1])))
    # k = 500
    # flips = 65
    # res = net.randomTests(k,flips)
    # print(flips,'bits flipped')
    # print('Correct: ',str(res)+'/'+str(k)+' = '+str(100*res//k)+' %')
    
    # for i in range(26):
        # print(i,net.testLetter(i))
    
    # print(net.examples)
    
    # net = majorityNetwork()
    # net.printAllWeights()
    # print(net.examples)
    # print('res',net.testInput([0,0,1]))
    # net.writeToFile()
    # net1 = Network([],[])
    # net1.initializeFromFile()
    # net1.printAllWeights()
    # print(net1.examples)