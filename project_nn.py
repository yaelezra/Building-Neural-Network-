
#importing packages
import numpy as np
import glob
import random
from matplotlib import pyplot as plt

### Loading training set

list_train = glob.glob('training/*.png')

train_size=256*2
trainingSet = np.zeros([train_size, 32*32])
traininglabels = np.zeros([train_size,1])

n=0
for image in list_train:
    im=plt.imread(image)
    [r,c]=np.shape(im)
    vect_t = np.reshape(im,[1,r*c]) 
    trainingSet[n,:] = vect_t
    if image[9:12] == 'pos': 
        traininglabels[n] = 1
    n += 1
            
### Loading validation set

list_valid = glob.glob('validation/*.png')

valid_size=156
validationSet = np.zeros([valid_size, 32*32])
validationlabels = np.zeros([valid_size,1])

n=0
for image in list_valid:
    im=plt.imread(image)
    [r,c]=np.shape(im)
    vect_v = np.reshape(im,[1,r*c]) 
    validationSet[n,:] = vect_v
    if image[11:14] == 'pos': 
        validationlabels[n] = 1
    n += 1


## Initializing weights and biases
def initializeWB(inputLayers,hLayers,outputLayers):
    np.random.seed(1) # Seed the random number generator
    W1 = np.random.randn(inputLayers, hLayers) 
    b1 = np.random.randn(hLayers,)
    W2 = np.random.randn(hLayers,outputLayers) 
    b2 = np.random.randn(outputLayers,)
    return W1, b1, W2, b2


## Activation function reLU
def relu(Z):
    return np.maximum(0,Z)

def dRelu(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z


## Activation function Sigmoid
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def dSigmoid(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))


##  loss function
def loss(y,yhat,flag):
    if flag=='Quad':
        return np.divide(np.sum((y - yhat)**2),2*len(y))
    elif flag=='log':
        smallnum = 0.0000000001
        return -1/len(y) * (np.sum(y*np.log(yhat + smallnum) + (1-y)*np.log(1-yhat + smallnum)))
  
    
## deriv of loss 
def dloss(y,yhat,flag):
    if flag=='Quad':
        return np.divide(np.sum((yhat - y)),len(y))
    elif flag=='log':
        return -1/len(y) * np.sum((y*(1-yhat)-(1-y)*yhat))


## Forward Propagation
def forward(x,w,b,flag):
    z = np.dot(x,w) + b
    if flag == 'relu':
        return relu(z)
    elif flag == 'sigmoid':
        return sigmoid(z)


## Back Propagation
def backPropagation(y,predsOutput,predsHidden,x,w1,w2,b1,b2,lr,loss_type):

    loss_output=dloss(y,predsOutput,loss_type) #loss of output
    
    delta_output=loss_output*dSigmoid(predsOutput) 

    hidden_loss=np.dot(delta_output,w2.T) #loss of hidden layer

    delta_hidden=hidden_loss*dRelu(predsHidden)
    
    delta_hidden=np.expand_dims(delta_hidden,axis=1)
    x=np.expand_dims(x,axis=1)
    delta_output=np.expand_dims(delta_output,axis=1)
    predsHidden=np.expand_dims(predsHidden,axis=1)

    #updating weights and biases

    dw1=np.dot(x,delta_hidden.T)
    db1=np.sum(dw1,axis=0)

    dw2=np.dot(predsHidden,delta_output)
    db2=np.sum(dw2,axis=0)

    w1=w1-lr*dw1
    b1=b1-lr*db1
    
    w2=w2-lr*dw2
    b2=b2-lr*db2

    return w1,b1,w2,b2


def accuracy(y,yhat):
    if y == np.round(yhat):
        return 1
    elif y!= np.round(yhat):
        return 0


###Running Neural Network### 


hiddenLayers=150
epochs=750
batches=128
lr=0.09

loss_type='Quad'

epochAccTrain = np.zeros([epochs,1])
epochLossTrain = np.zeros([epochs,1])


# initialization for the validation set
lossValid = np.zeros([valid_size,1])
accValid = np.zeros([valid_size,1])

epochAccValid = np.zeros([epochs,1])
epochLossValid = np.zeros([epochs,1])


w1,b1,w2,b2 = initializeWB(r*c,hiddenLayers,1) #input, hidden, output

for epoch in range(0,epochs):
    
    randlist = np.random.choice(range(0,511),batches,replace=False)
    batchLoss = np.zeros([batches,1])
    batchAcc = np.zeros([batches,1])
    trainingBatch = trainingSet[randlist]
    batchLabels = traininglabels[randlist]

    for batch in range(0,batches):
        x=trainingBatch[batch,:]
        y=batchLabels[batch]

        #forward propagation
        predsHidden = forward(x,w1,b1,'relu')
        predsOutput = forward(predsHidden,w2,b2,'sigmoid')
        
        #back propagation
        w1,b1,w2,b2=backPropagation(y,predsOutput,predsHidden,x,w1,w2,b1,b2,lr,loss_type)
        

        #loss and acuuracy
        batchLoss[batch] = loss(batchLabels[batch],predsOutput,loss_type)
        batchAcc[batch] = accuracy(batchLabels[batch],predsOutput)
        
   # computing the forward for the validation - to get the prediction for each w1,b1,w2,b2

    predsHiddenValid = forward(validationSet,w1,b1,'relu')
    predsOutputValid = forward(predsHiddenValid,w2,b2,'sigmoid')
    
   # computing the loss and accuracy for validation per epoch
    for valid in range (0,valid_size):
        lossValid[valid] = loss(validationlabels[valid],predsOutputValid[valid],loss_type)
        accValid[valid] = accuracy(validationlabels[valid],predsOutputValid[valid])

    epochLossTrain[epoch] = np.mean(batchLoss)
    epochAccTrain[epoch] = np.mean(batchAcc)

    epochLossValid[epoch] = np.mean(lossValid)
    epochAccValid[epoch] = np.mean(accValid)


# plots for loss
plt.plot(range(0,epochs),epochLossTrain)
plt.plot(range(0,epochs),epochLossValid)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])
plt.title('loss as a function of the epochs')
plt.show()

#plots for accuracy
plt.plot(range(0,epochs),epochAccTrain)
plt.plot(range(0,epochs),epochAccValid)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','validation'])
plt.title('accuracy as a function of the epochs')
plt.show()


#saving json
import json
import os

def make_json(W1, W2, b1, b2, id1, id2,id3,activation1, activation2, nn_h_dim, path_to_save):
    """
    make json file with trained parameters.
    W1: numpy arrays of shape (1024, nn_h_dim)
    W2: numpy arrays of shape (nn_h_dim, 1)
    b1: numpy arrays of shape (1, nn_h_dim)
    b2: numpy arrays of shape (1, 1)
    nn_hdim - number of neirons in hidden layer: int
    id1: id1 - str '0123456789'
    id2: id2 - str '0123456789'
    activation1: one of only: 'sigmoid', 'tanh', 'ReLU'
    activation2: one of only: 'sigmoid', 'tanh', 'ReLU'
    """
    trained_dict = {'weights': (W1.tolist(), W2.tolist()),
                    'biases': (b1.tolist(), b2.tolist()),
                    'nn_hdim': nn_h_dim,
                    'activation_1': activation1,
                    'activation_2': activation2,
                    'IDs': (id1, id2,id3)}   
    file_path = os.path.join(path_to_save, 'trained_dict_{}_{}_{}.json'.format(
        trained_dict.get('IDs')[0], trained_dict.get('IDs')[1],trained_dict.get('IDs')[2])
                             )
    with open(file_path, 'w') as f:
        json.dump(trained_dict, f, indent=4) 


make_json(w1, w2, b1, b2, '311334502', '324596857','316097567','ReLU', 'sigmoid', 150, 'C:/Users/Yael A/Desktop/image processing 2/project/ms_classification_project')

