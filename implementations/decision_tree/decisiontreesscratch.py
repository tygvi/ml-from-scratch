import numpy as np
class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion         
        self.max_depth = max_depth         
        self.min_samples_split = min_samples_split  
        self.root = None
        
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        if self.criterion == 'entropy':
          impurity = self._entropy(y)
       else:
          impurity = self._gini(y)
       return self

    def entropy(self,y):
        problist=[]
        tsum=0
        uniq=list(set(y))
        for i in range(len(uniq)):
            prob = np.sum(np.array(y) == uniq[i]) / len(y)
            problist.append(prob)
        for u in range(len(problist)):
            if problist[u] > 0
            ent=(-1)*problist[u]*np.log2(problist[u])
            tsum=tsum+ent
        return tsum
    
    def gini(self,y):
        problist=[]
        tsum=0
        uniq=list(set(y))
        for i in range(len(uniq)):
            prob = np.sum(np.array(y) == uniq[i]) / len(y)
            problist.append(prob)
        for u in range(len(problist)):
            if problist[u] > 0
            ginii=problist[u]*(1-problist[u])
            tsum=tsum+ginii
        return tsum
        
    def information_gain():
        pass
    def best_split():
        pass
    def build_tree():
        pass
    def predict():
        