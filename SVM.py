from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

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
    

def runSVM(parameterC, degreeP):
    (X, y) = readAbaloneData()
    (X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container) = dataStratification(X, y, 10)

    #Train and run SVM
    scoreList = []
    for i in range(len(X_Train_Container)):
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
        #SVM parameters
        clf = SVC(C=parameterC, kernel='poly', degree=degreeP, gamma='auto', coef0=0.0, tol=0.0001, cache_size=1024, max_iter=-1)
        #Train
        clf.fit(X_Train, y_Train)
        #Test
        scoreList.append(clf.score(X_Test, y_Test))
        #print "Test score ", i
        #print "--> ", scoreList[i]
    #end for
    #End Train and run MLP

    mean = 0
    for score in scoreList:
        mean += score

    mean = mean/len(scoreList)
    return mean

    #print "Mean: ", mean


def main():
    meanList = []
    print 250, "\t", 6, "\t", runSVM(250, 6)
    for C in range(20):
        for d in range(6):
            degreeP = d + 1
            paramC = 50*C + 300
            if(paramC == 0):
                paramC=1
            print paramC, "\t", degreeP, "\t", runSVM(paramC, degreeP)
    

if __name__ == "__main__":
    main();
