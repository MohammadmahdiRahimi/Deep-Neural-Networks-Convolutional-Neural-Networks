import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize (x):      #function for normalization
    return((x - min(x)) / (max(x) - min(x)))

def d_tanh(x):          #function of derivative tanh
    return (1 - np.tanh(x) ** 2)

def data_maker(num_of_data):        #this function make data which you can change number
    x = np.random.uniform(0, 2 * np.pi, num_of_data)
    y = np.random.uniform(0, 2 * np.pi, num_of_data)
    tag = np.sin(x + y)
    x = normalize(x)
    y = normalize(y)
    return x,y,tag

def split_data(x,y,tag):            #split data to validation , train and test with tags
    df = pd.DataFrame([x, y]).T
    data_train_with_valid = df.sample(frac=0.8) 
    data_test = df.drop(data_train_with_valid.index)
    data_valid = data_train_with_valid.sample(frac=0.2)
    data_train = data_train_with_valid.drop(data_valid.index)
    tag_train = tag[data_train.index]
    tag_test = tag[data_test.index]
    tag_valid = tag[data_valid.index]
    return data_train,data_test,data_valid,tag_train,tag_test,tag_valid

def predict( x,w1,w2,b1,b2):        #use weight to calculate output 
    x = x/1.0
    y1 = np.tanh(np.dot(x,w1) +b1.T)
    y = np.dot(y1, w2) + b2
    return [y, y1]



epo=5000            #number of iteration
num_of_data = 10000     #number of data
num_of_nodes = 15       #number of nodes
x,y,tag = data_maker(num_of_data)
data_train, data_test, data_valid, tag_train, tag_test, tag_valid = split_data(x,y,tag)
data_train = np.array(data_train)
tag_train = np.array(tag_train)

w1 = np.random.rand(2, num_of_nodes)        #initial weights
w2 = np.random.rand(num_of_nodes, 1)
b1 = np.random.rand(num_of_nodes, 1)
b2 = 1

train_error = np.zeros(epo)     #array of each iteration error for train data
test_error = np.zeros(epo)      #array of each iteration error for test data

for i in range(epo):        #update weights with gradient descent method
    for iter in np.random.randint(len(data_train), size=100):
        temp_train = data_train[iter]
        temp_tag = tag_train[iter]
        pred1,pred2 = predict(temp_train,w1,w2,b1,b2)
        error = temp_tag - pred1
        #gradient of weights
        delta_w1 = np.dot(temp_train.reshape(2, 1), error *d_tanh(np.dot(temp_train,w1) + b1.T) * w2.T)
        delta_w2 = np.dot(pred2.T, error)
        delta_b1 = error * d_tanh(np.dot(temp_train, w1) + b1.T) * w2.T
        delta_b2 = error
        #update weights
        w2 = w2 + 0.01 * delta_w2
        w1 = w1 + 0.01 * delta_w1
        b1 = b1 + 0.01 * delta_b1.T
        b2 = b2 + 0.01 * delta_b2
    #find predict with weights and calculate errror
    train_pred = predict(data_train,w1,w2,b1,b2)[0]
    test_pred = predict(data_test,w1,w2,b1,b2)[0]
    train_error[i] = ((train_pred.T - tag_train) ** 2).mean()
    test_error[i] = ((test_pred.T - tag_test) ** 2).mean()

#print errors of validation, train and test
valid_pred = predict(data_valid,w1,w2,b1,b2)[0]
valid_error = ((valid_pred.T - tag_valid) ** 2).mean()
print('               Error')
print('Train        ',train_error[-1])
print('Validation   ', valid_error)
print('Test         ',test_error[-1])
#plot errors of train and test
plt.plot(test_error)
plt.ylabel('Error')
plt.xlabel('iteration')
plt.show()
plt.plot(train_error)
plt.ylabel('Error')
plt.xlabel('iteration')
plt.show()
#next part which we should compare model with sine
xx = np.linspace(0, 2 * np.pi, 2000)
xx_normalize = normalize(xx)
yy = np.zeros(2000)
data = pd.DataFrame([xx_normalize, yy])
predicted = []
predicted = predict(data.T,w1,w2,b1,b2)[0]
predicted = np.array(predicted)
plt.plot(xx, np.sin(xx), label="Actual")
plt.plot(xx, predicted, label="Predicted")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()