import pandas as pd 
import numpy as np

class linear_regression_model:
    def __init__(self,help=False,math_breakdown=False) -> None:
        self.help=help
        self.math_breakdown=math_breakdown
        self.general_info=f"""          
        Linear regression is a regression technique that assumes that the data is linearly related to eachother.
        A few key terms that we will need to know are Mean, Variance, Covariance.

        Mean: Calculated as sum(inputs)/len(inputs). This gives us a single value that is the average of all data points.
        Variance: Calculated as sum((x-mean(x)**2)). Sum of squared difference from value and mean.
        Covariance: Calculated as sum((x(i) - mean(x)) * (y(i) - mean(y)). Describes how numbers change relationally to one another.

        Our goal with a simple linear regression is estimatinmg two coefficients: B0 and B1.

        """
        if self.help:
            print(self.general_info)
        self.steps=""
        self.trained=False


    def mean(self,values):
        means=np.mean(values)
        if self.math_breakdown:
            self.steps+=f"\nWe will find the mean of our input data (first few items:{values [:3]}) which gives us {means}"
        return means


    def variance(self,values,mean):
        var=sum([(x-mean)**2 for x in values])
        if self.math_breakdown:
            self.steps+=f"\nWe will find the variance of our input data (first few items: {values[:3]}) which gives us {var}"
        return var


    def covariance(self,x,mean_x,y,mean_y):
        covar=0.0
        self.steps+='\n\ncovariance steps:'
        self.itterative_covar=[] #store co variances for later graphing
        for i in range(len(x)):
            covar+=(x[i]-mean_x)*(y[i]-mean_y)
            self.itterative_covar.append(covar)
            if self.math_breakdown:
                self.steps+=f"\n    covariance now equals: {covar}"
        return covar
    
    def simple_dataset(self):
        self.dataset=[[1,1],[2,3],[4,3],[3,2],[5,5]]
        self.x=[row[0] for row in self.dataset]
        self.y=[row[1] for row in self.dataset]
    
    def train(self,dataset=None):
        if dataset is None:
            self.simple_dataset()
        else:
            self.dataset=dataset
        if isinstance(dataset, list):
            x=[row[0]for row in self.dataset]
            y=[row[1]for row in self.dataset]
        else:
            raise TypeError("Be sure you pass in a list of lists of data.")
        x_mean=self.mean(x)
        y_mean=self.mean(y)
        self.b1=self.covariance(x,x_mean,y,y_mean)/self.variance(x,x_mean)
        self.b0=y_mean-self.b1*x_mean
        if self.math_breakdown:
            self.steps+=f"\ntaking the covariance of X, X mean, Y, and Y mean divided by the variance of x and x mean to get our b1 score:{self.b1}"
            self.steps+=f"\nTaking the y_mean - b1 score and multiplying by x_mean to get b0 score:{self.b0}"
        self.trained=True            
        return [self.b0,self.b1]
    def predict(self,samples):
        if self.trained==False:
            # with 
            raise ValueError("Be sure to train your model before predicting.")
        preds=[self.b0 + self.b1 * i[0] for i in samples]
        return preds

    


if __name__=="__main__":
    c=linear_regression(help=True,math_breakdown=True)
    c.simple_dataset()
    c.train()
    print(c.predict([[1,3]]))   
    print(c.steps)
