from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import math
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

        print 'Classe\tAtributo\tMedia\t\t\tVariancia'
        for Sex in {'M', 'F', 'I'}:
                for Param in {'L', 'D', 'H', 'We', 'Sd', 'Va', 'Sl', 'R'}:
                        Variance[Sex][Param] /= count[Sex];
                        print  Sex + '\t' + Param+ '\t\t' , "%.7f" % Mean[Sex][Param], '\t\t', "%.7f" % Variance[Sex][Param]

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
                if (Y_test[i] == classe):
                        acerto = acerto +1
                else:
                        erro = erro+1            
        return float(acerto)/(acerto + erro)

def main():
	(X, y) = readAbaloneData()
	(X_Train_Container, y_Train_Container, X_Test_Container,y_Test_Container) = dataStratification(X, y, 10)
	#Train and run MLP
	scoreList = []
	for i in range(len(X_Train_Container)):
		#Get subsets
		X_Train = X_Train_Container[i]
		y_Train = y_Train_Container[i]
		X_Test = X_Test_Container[i]
		y_Test = y_Test_Container[i]
		#Scale
		#scaler = StandardScaler()
		#scaler.fit(X_Train)
		#X_Train = scaler.transform(X_Train)
		#X_Test = scaler.transform(X_Test)
		#MLP parameters
		score= classificadorBayesiano(X_Train, y_Train, X_Test, y_Test)
		
		#clf = MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic', solver='adam', learning_rate_init=0.001, max_iter=1000,tol=0.0001, verbose=False)
		#Train
		#clf.fit(X_Train, y_Train)
		#Test
		scoreList.append(score)
		print "Test score ", i
		print "--> ", scoreList[i]
	#end for
	#End Train and run MLP

	mean = 0
	for score in scoreList:
		mean += score

	mean = mean/len(scoreList)

	print "Mean: ", mean


if __name__ == "__main__":
	main();
