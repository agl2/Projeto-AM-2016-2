import pickle
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
d1 = [[1,2,4],[4,5,6],[7,8,9]]

train_index, test_index = skf.split( d1, ['a','b','c'])
print train_index, test_index 
