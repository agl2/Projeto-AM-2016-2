from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pickle

def readAbaloneData():
    f = open("abalone.data", "r")
    inputParameters = []
    outputClass = []
    for line in f:
        strArray = line.rstrip().split(",")
        inputParameters.append(map(float, strArray[1:9]))
        outputClass.append(strArray[0])
    return (inputParameters, outputClass)
#end readAbaloneData

def dataStratification(X, y, K):
    ##K-fold stratification
    skf = StratifiedKFold(n_splits=K)
    X_Train_Container = []
    X_Test_Container = []
    y_Train_Container = []
    y_Test_Container = []
    
    for train_index, test_index in skf.split(X, y):
        
        #print("TRAIN:", len(train_index), "TEST:", len(test_index))
        X_Train = []
        y_Train = []
        X_Test = []
        y_Test= []
        
        for i in train_index:
            X_Train.append(X[i])
            y_Train.append(y[i])

        for i in test_index:
            X_Test.append(X[i])
            y_Test.append(y[i])

        X_Train_Container.append(X_Train)
        y_Train_Container.append(y_Train)
        X_Test_Container.append(X_Test)
        y_Test_Container.append(y_Test)
    #end for

    return  (X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container)
#end dataStratification
    

def runMLP(hd_layers_p = (10, ), activation_p = 'logistic' , solver_p = 'adam', learn_rate_p = 0.001, early_stopping_p = False, momentum_p = 0.9 ,max_iter_p = 1000 ,weight_init_range = [-1, 1]):
    
    k_fold_lenght = 10
    scoreList = []
    MLPs = []

    (X, y) = readAbaloneData()
    (X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container) = dataStratification(X, y, k_fold_lenght)

    for i in range(k_fold_lenght):
        mlp = MLPClassifier(hidden_layer_sizes=hd_layers_p , activation=activation_p, solver=solver_p, learning_rate_init=learn_rate_p, max_iter=max_iter_p, momentum = momentum_p )
        
        #Get subsets
        X_Train = X_Train_Container[i]
        y_Train = y_Train_Container[i]
        X_Test = X_Test_Container[i]
        y_Test = y_Test_Container[i]
        #Scale
        scaler = StandardScaler()
        scaler.fit(X_Train)
        X_Train = scaler.transform(X_Train)
        X_Test = scaler.transform(X_Test)
        #Train
        mlp.fit(X_Train, y_Train)
        #print "Weight matrix", mlp.coefs_

        #Test
        scoreList.append(mlp.score(X_Test, y_Test))
        print "Test score ", i
        print "--> ", scoreList[i]   
    #end for
    #End Train and run MLP
    mean = 0
    for score in scoreList:
        mean += score

    mean = mean/len(scoreList)

    print "Mean: ", mean
    return mean

def main():
    #for i in range(50):
     #   neurons_1 = (i)*50 + 200
      #  print (neurons_1, ), "\t", runMLP(hd_layers_p = (neurons_1, ))
    #camada = (25,25,25,25,25)
    #for i in range(5):
        #camada = camada + (25,)
        #print camada, "\t", runMLP(hd_layers_p = camada)

    hdls = (25, )
    #for act in  ["identity", "logistic", "tanh", "relu"]:
     #   print act, "\t", runMLP(hd_layers_p = hdls, activation_p = act)

    act = 'tanh'
    #for sol in ['lbfgs', 'sgd', 'adam']:
        #print sol, "\t", runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol)

    sol = 'adam'
    #for sol in ['lbfgs', 'sgd', 'adam']:
        #print sol, "\t", runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol)

    #for div in range(5):
        #d = 10**(div)
        #learn_rate = 0.1/d
        #print learn_rate, "\t", runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol, learn_rate_p=learn_rate) 
    learn_rate = 0.001
    runMLP(hd_layers_p = hdls, activation_p = act, solver_p = sol, learn_rate_p=learn_rate)
    
                
if __name__ == "__main__":
    main();
