import numpy as np 
from collections import Counter

class decision_tree:
    def __init__(self) -> None:
        self.help=f"Decision trees are "
    def gini_index(self,groups,classes):
        n_instances=float(sum([len(group) for group in groups])) #total num of items in list 
        gini=0.0
        for group in groups:
            size=float(len(group))
            if size == 0: 
                continue
            score = 0.0
            for class_val in classes:
                p=[row[-1] for row in group].count(class_val)/size
                for row in group:
                    item=[row[-1]].count(class_val)/size
                score +=p*p
            gini += (1.0-score)*(size/n_instances)
        return gini


    def test_split(self,index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

        
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}
if __name__=="__main__":
    dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	[3.678319846,2.81281357,0],
	[3.961043357,2.61995032,0],
	[2.999208922,2.209014212,0],
	[7.497545867,3.162953546,1],
	[9.00220326,3.339047188,1],
	[7.444542326,0.476683375,1],
	[10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
    group=[[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
    group2=[[[1, 0], [1, 0]], [[1, 1], [1, 1]]]
    i=decision_tree()
    i.gini_index(groups=group,classes=[0,1])
    i.gini_index(groups=group2,classes=[0,1])
    # i.test_split()
    i.get_split(dataset=dataset)