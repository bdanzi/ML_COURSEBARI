#########################################################################################################
##  One hidden layer Neural Network with BACKPROPAGATION in action.   MAR2022 - Brunella D'Anzi        ##
#########################################################################################################
##  run under  ipython or python3 with the command 'python NN_backpropagation_bdanzi.py'
##  A h+p nodes, p input nodes, h hidden nodes, o=1 output, neural net fed samples data
##  point to learn the multiplication between p factors (not too much large otherwise we obtain too much large weights).
## To adjust the weights, we need to backprop from the NN output y, for the input x.  
## We take the partial derivatives of Loss with regard to weights dL/dWih and dL/dWho,
## using the Chain Rule:
## dL/dO = d(0.5 * (O(p) - targ(p))**2)/dO = ( O(p) - target(p))
## Then, compute the proper dL/dO * dO/dW.
## To be more precise, to adjust weights for the loss reduction, 
## we take the first derivative of the NN output
## with respect to weights dO/dwh, dO/dah, dO/db, dO/duih

from math import exp, tanh
from matplotlib import pyplot as plt
from platform import java_ver
import random, os


###################################################
##### NN Hyper-parameters and weights declaration #
###################################################

random.seed(10) # fixed a seed for reproducibility of the results
isAskUser = False
if isAskUser == True:
    epochs = int(input('Epochs:'))   # number of application of the weights update rule to all the examples of the training set
    eta = float(input('Rate learning (eta):'))   #  eta sets the rate of learning feedback
    samples = int(input('NSamples:')) # number of examples
    epsilon = float(input('Change in loss function: (epsilon)')) # percentage by which the mean loss function should change from one epoch to the previous one
    p = int(input('NInputs:')) # number of input nodes
    h = int(input('NHidden:')) # number of hidden nodes
    o = int(input('NOutput:')) # number of output nodes
    af_name = input('Activation function per each node:')
else:
    epochs = 200   # number of application of the weights update rule to all the examples of the training set
    eta = 0.0001   #  eta sets the rate of learning feedback
    samples = 20000 # number of examples
    epsilon = 0.0001 # percentage by which the mean loss function should change from one epoch to the previous one
    p = 2 # number of input nodes
    h = 50 # number of hidden nodes
    o = 1 # number of output nodes

x = list() # input "data" for each sample
y_true = list() # target for each sample
wh = list() # h weights of the hidden layer
ah = list() # h hidden biases
uih = list() # pxh input-hidden weights
zh = list() # z argument of the activation function
af_name = 'relu' # activation function used per each node


###################################################
##### Function definition for generalization  ##### 
###################################################

# input layer activation function: just pass input along
def I(p):
    return p

def targ(inputs,p):    ## the target is the product of all the inputs for each sample. If p=2 and the x1=x2, the it becomes x**2 and so on.
    y_true = 1 
    for i in range(p):
        y_true = y_true * inputs[i]
    return y_true
   
# error/loss function

def MSE(y_true,y_NN_opinion):   ## Using the very common Mean Square Error function for loss.
    loss = 0.5 * (y_NN_opinion - y_true[0])**2 ## I can have more than one input (p), but the loss function is computed w.r.t. first one
    return loss


# Generate random pairs (x,y) where x are the features and y true labels for each sample

def generate_inputs(inputs,y_true,p,samples):
    print('Generation of', p, 'input "data" and targets for', samples,'events')
    for s in range(samples):
        inputs.append([])
        y_true.append([])
        for i in range(p):
            inputs[s].append(round(random.uniform(1, 5),1)) # uniform generation from 1 to 5
            #inputs[s].append(random.gauss(0, 1)) # gaussian generation with mu = 0, sigma = 1
        for m in range(o):
            y_true[s].append(targ(inputs[s],p))
        #print('Input "data"',i,'Value',inputs[i], 'Target', y_true[i])
    return inputs

# Generate random 1-line weights

def generate_weights(weights,p):
    print('Generation of', p, 'Weights')
    for i in range(p):
        weights.append(round(random.uniform(0.001, 0.999),3))
        #print('Weight"',i,'Value',weights[i])
    return weights

# Generate random matrices weights

def generate_2D_weights(weights,p,h):
    print('Generation of', p, 'x',h,'Weights')
    for i in range(p):
        weights.append([])
        for l in range(h):
            weights[i].append(round(random.uniform(0.001, 0.999),3)) # generation of weights uniformly from 0.001 to 0.999 with 3 decimal digits
            #print('Weight',i,'Value',weights[i][l])
    return weights

# Compute the argument for the activation function, that is zh = ah + Sum{0,P-1}uih*xi for the s-th sample

def activation_function_argument(zh,ah,uih,x,s):
    zh = []
    for l in range(len(ah)):
        zh.append(ah[l])
        for i in range(len(x[s])):
            #if i == 0:
            zh[l] = zh[l] + uih[i][l]*I(x[s][i])
            #zh[l] = zh[l] + uih[i][l]*I(x[s][0])
    return zh

#############################################
###### Activation function definition #######
#############################################

def af(x,afname):
    if afname == 'relu':
        if x <= 0 :
            output_af = 0.
        else :
            output_af = x
    if afname == 'tanh':
        output_af = tanh(x)
    return output_af

#######################################################################
######## First Derivative of the Activation Function definition #######
#######################################################################

def deriv_af(x,dafname):
    if dafname == 'relu':
        if x < 0 or x == 0:
            output_daf = 0.
        else:
            output_daf = 1.
    if dafname == 'tanh':
        output_daf = 1 - (af(x,'tanh'))**2
    return output_daf

# Compute the output of each hidden layer Perceptron

def output_Perceptron(l, zh, wh):
    output_neuron = 0.
    output_neuron = wh[l] * af(zh[l],af_name)
    return output_neuron

# Compute the output of the hidden layer

def output_hidden_layer(zh,wh):
    output_hl = 0.
    for l in range(len(zh)):
       output_hl = output_hl + output_Perceptron(l,zh,wh)
    return output_hl

# output of the 1-hidden layer NN: multiply by weight Wh and add bias b; this then
# becomes the NN output y.
# For each sample the applied formula is:
# O(x,W) = b + Sum{0,h-1}(wh*f(ah+Sum{0,P-1}uih*xi))

def output_NN(bias,zh,wh):
    output_NN_value = 0.
    output_NN_value = bias + output_hidden_layer(zh,wh)
    return output_NN_value

# First derivative of the MSE loss function w.r.t. NN output
def dl_dO (y_NN_opinion,y_true,s):
    dl_over_dO = y_NN_opinion - y_true[s][0]
    return dl_over_dO

# Update procedure for weights 
def weights_updates(whl,eta,dl_over_dO,dO_over_w):
    whl = whl - eta * dl_over_dO * dO_over_w
    return whl


###############################################
######## WEIGHTS RANDOM initialization ########
###############################################

bias = round(random.uniform(0.001, 0.999),3) #value of the output bias
generate_weights(wh,h) #hidden weights
#print(h,'Hidden weights wh', wh)

generate_weights(ah,h)
#print(h,'Hidden weights ah', ah)

generate_2D_weights(uih,p,h)
#print(p,'x',h,'weights uih', uih)

#####################################################
########## Input and Target data generation #########
#####################################################

generate_inputs(x,y_true,p,samples)
#print(p, 'Inputs for all',samples,'events:', x, p, 'Targets', y_true)

############################################################
######### TRAINING PROCEDURE with BACKPROPAGATION ##########
############################################################

print('!!!!!!!!!!!!TRAINING phase!!!!!!!!!!')
tot_loss_old = 0 # buffer for the loss
current_epochs = list()
samples_test = 1000
x_test = list() # input "data" for each testing sample
y_true_test = list() # target for each testing sample
y_pred_train = list()
generate_inputs(x_test,y_true_test,p,samples_test)
average_loss_test = list()
average_loss = list()

for e in range(epochs):
    tot_loss = 0 # loss computed for the current epoch
    tot_loss_test = 0
    dl_over_dO = 0
    dO_over_duih = list()
    dO_over_dah = list()
    dO_over_dwl = list()
    dO_over_db = list()
    for i in range(p): #loop over p inputs for updating uih weights
        dO_over_duih.append([])
        for l in range(len(wh)):
            dO_over_duih[i].append(0)
    for l in range(len(wh)):
        dO_over_dah.append(0)
        dO_over_dwl.append(0)    
    dO_over_db = 0
    for s in range(samples_test):
        zh = activation_function_argument(zh,ah,uih,x_test,s)
        y_NN_opinion_val = output_NN(bias,zh,wh)
        if s < 5:
            print("VALIDATION phase:","Epoch",e, "Sample",s,"Inputs", x_test[s],"True label", "{:.2f}".format(y_true_test[s][0]),"Predicted label","{:.2f}".format(y_NN_opinion_val),"Event Mean Square Error","{:.4f}".format(MSE(y_true_test[s],y_NN_opinion_val)))
        tot_loss_test = tot_loss_test + MSE(y_true_test[s],y_NN_opinion_val)
    for s in range(samples):
        zh = activation_function_argument(zh,ah,uih,x,s)
        y_NN_opinion = output_NN(bias,zh,wh)
        if s < 5:
            print("Training phase:",'Epoch',e,'Sample',s,'Inputs',x[s],'True label',"{:.2f}".format(y_true[s][0]),'Predicted label',"{:.2f}".format(y_NN_opinion),'Event MSE Loss', "{:.4f}".format(MSE(y_true[s],y_NN_opinion)))
        # Store the Mean Square Error (MSE) loss function for each sample and each epochs 
        tot_loss = tot_loss + MSE(y_true[s],y_NN_opinion)
        # Check if, for one epoch w.r.t. previous one, the loss function is changed more than epsilon for the sample under consideration.
        # If so, continue to update the weights and print the y_predicted. Otherwise, just print the results the first time you failed 
        # the condition for that sample.
        dl_over_dO = dl_over_dO + dl_dO(y_NN_opinion,y_true,s)
        for i in range(p): #loop over p inputs for updating uih weights
            for l in range(len(wh)): #loop also over the uih sizes
            # Dropout implementation, Not updating some weights 
                if random.uniform(0, 1) > 0.:
                    dO_over_duih[i][l] = dO_over_duih[i][l] + wh[l] * x[s][i] * deriv_af(zh[l],af_name)
        for l in range(len(wh)): # loop over the ah and wh size for updating
            if random.uniform(0, 1) > 0.: # Dropout implementation, Not updating some weights #
                dO_over_dah[l] = dO_over_dah[l] + 1 * wh[l] * deriv_af(zh[l],af_name)
            if random.uniform(0, 1) > 0.:
                dO_over_dwl[l] = dO_over_dwl[l] + af(zh[l],af_name)
        if random.uniform(0, 1) > 0.:
            dO_over_db = dO_over_db + 1
    for i in range(p): #loop over p inputs for updating uih weights
        for l in range(len(wh)): #loop also over the uih sizes
            uih[i][l] = weights_updates(uih[i][l],eta,dl_over_dO/samples,dO_over_duih[i][l]/samples)
    for l in range(len(wh)): #loop also over the uih sizes
        ah[l] = weights_updates(ah[l],eta,dl_over_dO/samples,dO_over_dah[l]/samples)
        wh[l] = weights_updates(wh[l],eta,dl_over_dO/samples,dO_over_dwl[l]/samples)
    bias = weights_updates(bias,eta,dl_over_dO/samples,dO_over_db/samples)
    average_loss_test.append(tot_loss_test/samples_test)
    average_loss.append(tot_loss/samples)
    current_epochs.append(e) 
    print('Average Validation loss in the i-th epoch')
    print(average_loss_test)
    print('Average Training loss in the i-th epoch')
    print(average_loss)
    # Plot the MSE loss function for training and test data sets 
    plt.clf()
    plt.plot(current_epochs, average_loss, label = "Training data set")
    plt.plot(current_epochs, average_loss_test, label = "Validation data set")
    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Mean Square Error')
    # giving a title to my graph
    plt.title('Training and Val loss function on Training and Val samples')
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.savefig("loss_train_val.png", dpi=300)
    # Check for changes on the mean loss function
    if (abs(tot_loss/samples - tot_loss_old/samples) < epsilon and e>0):
        print("TRAINING STOPPED by Loss function: Epoch",e,'Total mean training loss function', "{:.4f}".format(tot_loss/samples), 'Previous training loss function',"{:.4f}".format(tot_loss_old/samples))#,'Weights wh',wh,'Weights ah',ah,'Weights uih',uih,'Weight bias',bias)
        break
    tot_loss_old = tot_loss
        

#samples_holdback = int(input('Number of samples for the TEST (Holdback) phase:'))
samples_holdback = 300
x_holdback = list() # input "data" for each testing sample
y_true_test = list() # target for each testing sample
y_pred_holdback = list()
generate_inputs(x_holdback,y_true_test,p,samples_holdback)
for s in range(samples_holdback):
    zh = activation_function_argument(zh,ah,uih,x_holdback,s)
    y_NN_opinion_holdback = output_NN(bias,zh,wh)
    y_pred_holdback.append(y_NN_opinion_holdback)
    print("Test phase DONE!:", "Sample",s,"Inputs", x_holdback[s],"True label", "{:.2f}".format(y_true_test[s][0]),"Predicted label","{:.2f}".format(y_NN_opinion_holdback),"Mean Square Error","{:.2f}".format(MSE(y_true_test[s],y_NN_opinion_holdback)))
