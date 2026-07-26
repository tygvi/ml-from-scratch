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
        
    def information_gain(self,y,left_y,right_y):
        if self.criterion=="entropy":
           parent_impurity=self.entropy(y)
        else:
           parent_impurity=self.gini(y)
        if self.criterion=="entropy":
           left_impurity=self.entropy(left_y)
           right_impurity=self.entropy(right_y)
        else:
            left_impurity=self.gini(left_y)
            right_impurity=self.gini(right_y)
            
        left_w=len(left_y)/len(y)
        right_w=len(right_y)/len(y)

        weighted_impurity=(left_w*left_impurity)+(right_w*right_impurity)
        gain=parent_impurity-weighted_impurity
        return gain
    
    def best_split(self, X, y):
      best_gain = -float("inf")
      best_threshold = None
      values = np.sort(np.unique(X))
      thresholds = (values[:-1] + values[1:]) / 2
      for threshold in thresholds:
        left_mask = X < threshold
        right_mask = X >= threshold
        left_y = y[left_mask]
        right_y = y[right_mask]
        if len(left_y) == 0 or len(right_y) == 0:
            continue
        gain = self.information_gain(y, left_y, right_y)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    return best_threshold, best_gain

    def build_tree():
    def predict():
        
        