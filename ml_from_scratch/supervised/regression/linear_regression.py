import pandas as pd 
import numpy as np

class linear_regression:
    def __init__(self,help=False) -> None:
        self.help=help
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


    def mean(self,values):
        means=np.mean(values)
        self.steps+=f"\nWe will find the mean of our input data which gives us {means}"
        return means


    def variance(self,values):
        var=np.variance(values)
        self.steps+=f"\nWe will find the variance of our input data which gives us {var}"
        return var


    def covariance(self,x,mean_x,y,mean_y):
        covar=0.0
        for i in range(len(x)):
            covar+=(x[i]-mean_x)*(y[i]-mean_y)
        return covar
    
    def simple_dataset(self):
        self.dataset=[[1,1],[2,3],[4,3],[3,2],[5,5]]
        self.x=[row[0] for row in self.dataset]
        self.y=[row[1] for row in self.dataset]

    


if __name__=="__main__":
    c=linear_regression(True)