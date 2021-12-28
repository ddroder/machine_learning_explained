import numpy as np
import sys
class train_test_split:
    def __init__(self,x,y,train_size=.75,test_size=.25):
        self.x=x
        self.y=y
        self.train_size=train_size
        self.test_size=test_size
        # return self.train_test_split(x,y,train_size,test_size)
    def split_data(self):
        split_i = len(self.y) - int(len(self.y) // (1 / self.test_size))
        X_train, X_test = self.x[:split_i], self.x[split_i:]
        y_train, y_test = self.y[:split_i], self.y[split_i:]

        return X_train, X_test, y_train, y_test
        


if __name__=="__main__":
    x=[1,2,3,4,5,6,7,8,9,10]
    y=[1,2,3,4,5,6,7,8,9,10]
    c=train_test_split(x,y)
    xtrain,xtest,ytrain,ytest=c.split_data()
    print(xtrain)

