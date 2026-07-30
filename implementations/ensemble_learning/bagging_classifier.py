import numpy as np
from collections import Counter

class Node:
    def __init__(self, threshold=None, left=None, right=None, value=None):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2): #When called, it will tell the function which impurity to apply
        self.criterion = criterion #       
        self.max_depth = max_depth #It limits the maximum depth of the tree from the root to any of the leaves, preventing it from becoming overly complex        
        self.min_samples_split = min_samples_split #A node must contain at least this much samples before it is allowed to split. 
        self.root = None

        
# Train the Decision Tree on the given dataset.
# The input data is converted to NumPy arrays, after which the recursive build_tree() function constructs the entire tree.
# Returns self to support method chaining.
    
    def fit(self,X,y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root=self.build_tree(X,y)
        return self

    # Calculates the entropy of a node.
    # Lower entropy indicates a purer node with fewer mixed classes.
    def entropy(self,y):
        problist=[]
        tsum=0
        uniq=np.unique(y)
        for i in range(len(uniq)):
            prob = np.sum(np.array(y) == uniq[i]) / len(y)
            problist.append(prob)
        for u in range(len(problist)):
            if problist[u] > 0:
               ent=(-1)*problist[u]*np.log2(problist[u])
               tsum=tsum+ent
        return tsum
        
    # Calculates the Gini impurity of a node.
    # A lower Gini score means the node contains mostly one class.
    def gini(self,y):
        problist=[]
        tsum=0
        uniq=np.unique(y)
        for i in range(len(uniq)):
            prob = np.sum(np.array(y) == uniq[i]) / len(y)
            problist.append(prob)
        for u in range(len(problist)):
            if problist[u] > 0:
               ginii=problist[u]*(1-problist[u])
               tsum=tsum+ginii
        return tsum

    # Calculates the reduction in impurity after a split.
    # The parent impurity is compared against the weighted impurity
    # of the left and right child nodes. A higher information gain
    # indicates a better split.
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
        
    def build_tree(self, X, y, depth=0):
      if self.max_depth is not None and depth >= self.max_depth:
        majority = Counter(y).most_common(1)[0][0]
        return Node(value=majority)
      if len(y) < self.min_samples_split:
        majority = Counter(y).most_common(1)[0][0]
        return Node(value=majority)
      if len(set(y)) == 1:
        return Node(value=y[0])
      threshold, gain = self.best_split(X, y)
      if gain <= 0:
        majority = Counter(y).most_common(1)[0][0]
        return Node(value=majority)
      
      left_X = []
      left_y = []
      right_X = []
      right_y = []

      for i in range(len(X)):
        if X[i] < threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
      left_X = np.array(left_X)
      left_y = np.array(left_y)
      right_X = np.array(right_X)
      right_y = np.array(right_y)
      left_child = self.build_tree(left_X, left_y, depth + 1)
      right_child = self.build_tree(right_X, right_y, depth + 1)
      return Node(
        threshold=threshold,
        left=left_child,
        right=right_child
    )
     
     # Predicts the class labels for all input samples by
     # traversing the trained Decision Tree.
    def predict(self, X):
       X = np.asarray(X)
       predictions = []
       for x in X:
        predictions.append(self._predict(x, self.root))
       return np.array(predictions)
    
    def _predict(self, x, node):   
      if node.value is not None:
        return node.value
      if x < node.threshold:
        return self._predict(x, node.left)
      else:
        return self._predict(x, node.right)
        
        
class RandomForestClassifier:
    def __init__(self,n_estimators=100
                 max_depth=None,
                 criterion="gini"
                 min_samples_split=2,
                 max_features=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
    


    def fit():
        self.trees=[]
        for _ in range(self.n_estimators):
            X_boot,y_boot=self._bootstrap_sample(X,y)
            tree=decisiontreeclassifier(max_depth=self.max_depth,
                                        criterion=self.criterion,
                                        min_samples_split-self.min_samples_split,)
            tree.fit(X_boot,y_boot)
            self.trees.append(tree)
    return self

    def _bootstrap_sample(self,X,y):
        n_samples=X.shape[0]
        indices=np.random.choice(n_samples,size=n_samples,replace=True)
        X_boot=X[indices]
        y_boot=y[indices]
        return X_boot,y_boot
    def _majority_vote(self,prediction)
        values,counts=np.unique(predictions,return_counts=True)
        return values[np.argmax(counts)]
    def predict(self,X):
        predictions=[]
        for tree in self.trees:
            predictions.append(tree.predict(X))
        predictions=np.transpose(predictions)
        final_predictions=[]
        for sample_predictions in predictions:
            final_predictions.append(
            self._majority_vote(sample_predictions)
        )
    return np.array(final_predictions)
    
