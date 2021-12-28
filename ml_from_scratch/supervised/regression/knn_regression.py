import numpy as np 
# ml_from_scratch/ml_from_scratch/utils    
class knn_regressor:
    def __init__(self,distance='eucledian',math_breakdown=False):
        self.steps=f""
        self.distance=distance
        self.math_breakdown=math_breakdown
        pass
    def scaling(self,xtrain,ytrain,xtest=None):
        self.mu=np.mean(xtrain,0)
        self.sigma=np.std(xtrain,0)
        self.xtrain=(xtrain - self.mu)/self.sigma
        self.ytrain=(ytrain-self.mu)/self.sigma
        if self.math_breakdown:
            self.steps+=f"\nto scale the data, we will do a standard scaling (making the mean 0 and variance 1). We will subtract the mean from our training data and divide by the standard deviation."
            self.steps+=f"\nWe will repeat the above process to the test data, making sure to use the training mean and standard deviation (since we wont know what the test values are)."
        if xtest is not None:
            self.xtest=(xtest-self.mu)/self.sigma
    def train(self,xtrain,ytrain):
        if self.math_breakdown:
            self.steps+=f"\nWhat we do now is calculate the mean and standard deviation of our scaled data.\nScaling our data is very important for knn models, so we do it natively."
        self.y_mu=np.mean(ytrain)
        self.y_sigma=np.std(ytrain)
        # pass
    def predict(self,x,y,neighbors=2):
        if self.math_breakdown:
            self.steps+=f"\nWe will now generate predictions. To do this, we will use the mean and standard deviation of our y train data.\nwe will compute the {self.distance} distance of the features now."
        self.distance=self.get_distance(metric=self.distance,x=x,y=y,neighbors=neighbors)
    def get_distance(self,metric,x,y,neighbors=10):
        VALID_METRICS=['eucledian','manhattan']
        if metric not in VALID_METRICS:
            raise ValueError(f"Please select a valid distance metric:{VALID_METRICS}")
        if metric == "eucledian":
            preds=[]
            for row in range(len(x)):
                distance=self.eucledian_distance(x[row],y[row])
                pred=y[np.argsort(distance,axis=0)[:neighbors].mean()*self.y_sigma+self.y_mu]
                preds.append(distance)
            return preds


    def eucledian_distance(self,p1,p2):
        euc_distance=np.sqrt(np.sum((p1-p2)**2))
        if self.math_breakdown:
            self.steps+=f"\neucledian distance calc between points ({p1,p2}): {euc_distance}"
        return euc_distance
    def manhattan_distance(self,p1,p2):
        man_distance=np.sum(np.abs(p1,p2))
        if self.math_breakdown:
            self.steps+=f"\nmanhattan distance calc between points ({p1,p2}): {man_distance}"
        return man_distance
    
    

        


if __name__=="__main__":
    # print("im alive")
    x=[1,2,3,4]
    y=[5,6,7,8]
    c=knn_regressor(math_breakdown=True)
    c.train(x,y)
    c.predict(x,y)
    # print(c.distance)
    