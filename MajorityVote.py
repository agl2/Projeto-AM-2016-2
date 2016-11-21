from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pickle
import math

def take_mean_variance_Priori (X_train, Y_train):
        count = {'M': 0, 'F': 0, 'I': 0}
        Mean =  {'M': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                 'F': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                 'I': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                }
        
        Variance =  {
                        'M': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                        'F': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                        'I': {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0},
                        }
        P =  {
                        'M': 0,
                        'F': 0,
                        'I': 0,
                        }
        
        
        for i in range(len(Y_train)):
                        count[Y_train[i]] = count[Y_train[i]] + 1

        P['M'] = float(count['M'])/(count['M'] + count['F'] + count['I']);
        P['F'] = float(count['F'])/(count['M'] + count['F'] + count['I']);
        P['I'] = float(count['I'])/(count['M'] + count['F'] + count['I']);

                
        for i in range(len(X_train)):
                Mean[Y_train[i]]['L']  += X_train[i][0];
                Mean[Y_train[i]]['D']  += X_train[i][1];
                Mean[Y_train[i]]['H']  += X_train[i][2];
                Mean[Y_train[i]]['We'] += X_train[i][3];
                Mean[Y_train[i]]['Sd'] += X_train[i][4];
                Mean[Y_train[i]]['Va'] += X_train[i][5];
                Mean[Y_train[i]]['Sl'] += X_train[i][6];
                Mean[Y_train[i]]['R']  += X_train[i][7];

        for Sex in {'M', 'F', 'I'}:
                for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                        Mean[Sex][Param] /= count[Sex];

        for i in range(len(X_train)):
                Variance[Y_train[i]]['L']  += (X_train[i][0] - Mean[Y_train[i]]['L'])*(X_train[i][0] - Mean[Y_train[i]]['L']);
                Variance[Y_train[i]]['D']  += (X_train[i][1] - Mean[Y_train[i]]['D'])*(X_train[i][1] - Mean[Y_train[i]]['D']);
                Variance[Y_train[i]]['H']  += (X_train[i][2] - Mean[Y_train[i]]['H'])*(X_train[i][2] - Mean[Y_train[i]]['H']);
                Variance[Y_train[i]]['We'] += (X_train[i][3] - Mean[Y_train[i]]['We'])*(X_train[i][3] - Mean[Y_train[i]]['We']);
                Variance[Y_train[i]]['Sd'] += (X_train[i][4] - Mean[Y_train[i]]['Sd'])*(X_train[i][4] - Mean[Y_train[i]]['Sd']);
                Variance[Y_train[i]]['Va'] += (X_train[i][5] - Mean[Y_train[i]]['Va'])*(X_train[i][5] - Mean[Y_train[i]]['Va']);
                Variance[Y_train[i]]['Sl'] += (X_train[i][6] - Mean[Y_train[i]]['Sl'])*(X_train[i][6] - Mean[Y_train[i]]['Sl']);
                Variance[Y_train[i]]['R']  += (X_train[i][7] - Mean[Y_train[i]]['R'])*(X_train[i][7] - Mean[Y_train[i]]['R']);

        #print 'Classe\tAtributo\tMedia\t\t\tVariancia'
        for Sex in {'M', 'F', 'I'}:
                for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                        Variance[Sex][Param] /= count[Sex];
                        #print  Sex + '\t' + Param+ '\t\t' , "%.7f" % Mean[Sex][Param], '\t\t', "%.7f" % Variance[Sex][Param]

        return Mean, Variance, P

def p_xk_given_wk(mean, variance, Xk):
        d = 8
        Result= {'M':0, 'F':0, 'I':0}
        for s in {'M', 'F', 'I'}:
                mult_variance = 1
                exponent = 0;

                for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                        mult_variance *= variance[s][Param]
                
                for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                        #print s,' ', Param, ' parcial= ', (Xk[Param] - mean[s][Param])*(Xk[Param] - mean[s][Param])/variance[s][Param], '\n'
                        if  (   Xk[Param] > max(    mean['M'][Param],
                                                                        mean['F'][Param],
                                                                        mean['I'][Param])
                                                                +5*max( variance['M'][Param],
                                                                        variance['F'][Param],
                                                                        variance['I'][Param])
                                        or  Xk[Param] < min(    mean['M'][Param],
                                                                        mean['F'][Param],
                                                                        mean['I'][Param])
                                                                -5*max( variance['M'][Param],
                                                                        variance['F'][Param],
                                                                        variance['I'][Param])                           
                                ):
                                exponent=0
                        else:
                                exponent += (Xk[Param] - mean[s][Param])*(Xk[Param] - mean[s][Param])/variance[s][Param]
                exponent *= -1/2

                #print s, ' ', mult_variance,' ', exponent
                Result[s] = ((2*math.pi)**(-d/2))*(mult_variance**(-0.5))*math.exp(exponent);
                #print "P( ", Xk, " | ", s, " ) = ", Result[s], '\n'
        return  Result


def p_wk_given_xk(mean, variance, p, Xk):
        p_xk = 0
        p_xk_wk = p_xk_given_wk(mean, variance, Xk)
        Result= {'M':0, 'F':0, 'I':0}
        for s in {'M', 'F', 'I'}:
                p_xk += p_xk_wk[s]*p[s]
        for s in {'M', 'F', 'I'}:
                if(p_xk!=0):
                        Result[s] = p_xk_wk[s]*p[s]/p_xk
                else:
                        print "ERRO"
                        for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                                print Param + " " + str(Xk[Param]) + "\n"
                        sys.exit()
        return Result

def classificadorBayesiano(X_train, Y_train, X_test, Y_test):
        mean, variance, p = take_mean_variance_Priori (X_train, Y_train)
        Xk = {'L': 0, 'D':0, 'H':0, 'We':0, 'Sd':0, 'Va':0, 'Sl':0, 'R':0}
        acerto = erro = 0
        classeBAESY = []
        for i in range(len(X_test)):
                Xk['L']  = X_test[i][0];
                Xk['D']  = X_test[i][1];
                Xk['H']  = X_test[i][2];
                Xk['We'] = X_test[i][3];
                Xk['Sd'] = X_test[i][4];
                Xk['Va'] = X_test[i][5];
                Xk['Sl'] = X_test[i][6];
                Xk['R']  = X_test[i][7];
                result = p_wk_given_xk(mean, variance, p, Xk);
                if(result['M'] == max(result['M'], result['F'], result['I'])):
                     classe = 'M'
                elif(result['F'] == max(result['M'], result['F'], result['I'])):
                     classe = 'F'
                elif(result['I'] == max(result['M'], result['F'], result['I'])):
                    classe = 'I'

                classeBAESY.append(classe)
                if (Y_test[i] == classe):
                        acerto = acerto +1
                else:
                        erro = erro+1            
        return (float(acerto)/(acerto + erro), classeBAESY)

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
        
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
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

def loadMLP(k_fold_lenght):
    MLPs = []
    try:
        trainMLP = False
        fMLP = open("mlp.init", 'rb')
        for i in range(k_fold_lenght):
            mlp = pickle.load(fMLP)
            MLPs.append(mlp)
    except IOError as e:
        trainMLP = True
        print "Unable to open mlp.init file" #Does not exist OR no read permissions
    except EOFError as e:
        trainMLP = True
        print "Unable to read mlp.init file" #Does not exist OR no read permissions

    if(trainMLP):
        fMLP = open("mlp.init", 'wb')
        for i in range(k_fold_lenght):
            mlp = MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic', solver='adam', learning_rate_init=0.001, max_iter=1000,tol=0.0001, verbose=False)
            MLPs.append(mlp)
        

    return (MLPs, trainMLP, fMLP)
#end loadMLP

def loadSVM(k_fold_lenght):
    SVMs=[]
    try:
        trainSVM = False
        fSVM = open("svm.init", 'rb')
        for i in range(k_fold_lenght):
            svm = pickle.load(fSVM)
            SVMs.append(svm)
    except IOError as e:
        trainSVM = True
        print "Unable to open svm.init file" #Does not exist OR no read permissions
    except EOFError as e:
        trainSVM = True
        print "Unable to read svm.init file" #Does not exist OR no read permissions

    if(trainSVM):
        for i in range(k_fold_lenght):
            fSVM = open("svm.init", 'wb')
            svm = SVC(C=10.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.0001, cache_size=1024, class_weight=None, verbose=False, max_iter=-1)
            SVMs.append(svm)

    return (SVMs, trainSVM, fSVM)

#end loadSVM

def majorityVote(classeBAESY, classeSVM, classeMLP, y_Test):
    classeMajority = []
    classePossiveis = ['M', 'F', 'I']
    acerto = 0
    erro = 0
    #print classeBAESY
    for i in range(len(classeBAESY)):
        #print "Classificacao instancia ", i 
        voteCount = [0, 0, 0]
        for clf in [classeBAESY, classeSVM, classeMLP]:
            if(clf[i] == 'M'):
                voteCount[0] += 1
            elif(clf[i] == 'F'):
                voteCount[1] += 1
            elif(clf[i] == 'I'):
                voteCount[2] += 1

        #print "\tVotos: ", voteCount  

        if(voteCount[0] == voteCount[1] == voteCount[2]):
            classeMajority.append('M')
        else:
            classeMajority.append(classePossiveis[ voteCount.index(max(voteCount)) ] )

        #print "\tClassificada: ", classeMajority[i] 
        #print "\tReal: ", y_Test[i]       
        if(classeMajority[i] == y_Test[i]):
            acerto += 1.0
        else:
            erro += 1.0
            
    #print "Erro: ", erro
    #print "Acerto: ", acerto 
    return float(acerto/(acerto+erro))    
    

def main():
    
    k_fold_lenght = 10
    scoreListSVM = []
    scoreListMLP = []
    scoreListBAESY = []
    scoreListMAJ= []

    (X, y) = readAbaloneData()
    
    (X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container) = dataStratification(X, y, k_fold_lenght)
    
    (MLPs, trainMLP, fMLP) = loadMLP(k_fold_lenght)
    
    (SVMs, trainSVM, fSVM) =  loadSVM(k_fold_lenght)

    for i in range(k_fold_lenght):
        #Get classfier
        mlp = MLPs[i]
        svm = SVMs[i]
        #Get subsets
        X_Train = X_Train_Container[i]
        y_Train = y_Train_Container[i]
        X_Test = X_Test_Container[i]
        y_Test = y_Test_Container[i]
        #bayesiano
        (scoreBAESY, classeBAESY) = classificadorBayesiano(X_Train, y_Train, X_Test, y_Test)
        scoreListBAESY.append(scoreBAESY)
        #Scale
        scaler = StandardScaler()
        scaler.fit(X_Train)
        X_Train = scaler.transform(X_Train)
        X_Test = scaler.transform(X_Test)
        #Train
        if(trainMLP):
            print "Train MLP"
            mlp.fit(X_Train, y_Train)
            pickle.dump(mlp, fMLP)

        if(trainSVM):
            print "Train SVM"
            svm.fit(X_Train, y_Train)
            pickle.dump(svm, fSVM)
            
        #Test
        classeSVM = svm.predict(X_Test)
        classeMLP = mlp.predict(X_Test)

        scoreListMAJ.append( majorityVote(classeBAESY, classeSVM, classeMLP, y_Test) )
        scoreListSVM.append(svm.score(X_Test, y_Test))
        scoreListMLP.append(mlp.score(X_Test, y_Test))
        print "Test score ", i
        print "SVM --> ", scoreListSVM[i]
        print "MLP --> ", scoreListMLP[i]
        print "BAESY --> ", scoreListBAESY[i]
        print "MAJ --> ", scoreListMAJ[i]
    #end for
    #End Train and run MLP
    meanSVM = 0
    meanMLP = 0
    meanBAESY = 0
    meanMAJ = 0
    
    for score in scoreListSVM:
        meanSVM += score

    for score in scoreListMLP:
        meanMLP += score

    for score in scoreListBAESY:
        meanBAESY += score

    for score in scoreListMAJ:
        meanMAJ += score

    meanSVM = meanSVM/len(scoreListSVM)
    meanMLP = meanMLP/len(scoreListMLP)
    meanBAESY = meanBAESY/len(scoreListBAESY)
    meanMAJ = meanMAJ/len(scoreListMAJ)
    
    print "Mean SVM: ", meanSVM
    print "Mean MLP: ", meanMLP
    print "Mean BAESY: ", meanBAESY
    print "Mean MAJ: ", meanMAJ


if __name__ == "__main__":
    main();
