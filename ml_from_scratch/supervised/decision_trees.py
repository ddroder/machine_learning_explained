from collections import Counter
import numpy as np 


def entropy(y):
    hist=np.bincount(y) #num occurances of all class labels
    ps=hist / len(y)
    return -np.sum([p*np.log2(p) for p in ps if p > 0]) #need to check if less than zero since log for negative numbers in nan

class node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None) -> None:
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.value=value
        pass
    def is_leafnode(self):
        return self.value is not None
class decision_tree:
    def __init__(self,min_samples=2,max_depth=100,n_feats=None) -> None:
        self.min_samples=min_samples
        self.max_depth=max_depth
        self.n_feats=n_feats
        self.root=None
    def fit(self,x,y):
        #grow the tree out
        self.n_feats=x.shape[1] if not self.n_feats else min(self.n_feats,x.shape[1]) #make sure x is never bigger than n feats
        self.root=self._grow_tree(x,y)
        pass
    def predict(self,x):
        #traverse tree
        pass
    def _grow_tree(self,x,y,depth=0):
        n_samples,n_features=x.shape
        n_labels=len(np.unique(y))
        
        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples):
            leaf_value=self.most_common_label(y)
            return node(value=leaf_value)
        feat_idxs=np.random.choice(n_features,self.n_feats,replace=False)

        #greedy search of tree
        best_feat,best_thresh=self._best_criteria(x,y,feat_idxs)

    def _best_critera(self,x,y,feat_idxs):
        best_gain=-1
        split_idx,split_thresh=None,None
        for feat_idx in feat_idxs:
            x_col=x[:,feat_idx]
            thresholds=np.unique(x_col)
            for thresh in thresholds:
                gain=self._information_gain(y,x_col,thresh)
                if gain > best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_thresh=thresh
        return split_idx,split_thresh


    def _most_common_label(self,y):
        counter=Counter(y)
        most_common=counter.most_common(1)[0][0]
        return most_common


if __name__=="__main__":
    i=decision_tree()
    print(i.min_samples)