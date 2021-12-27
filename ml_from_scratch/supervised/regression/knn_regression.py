import numpy as np 

class knn_regressor:
    def __init__(self):
        self.steps=f""
        pass
    def data(self):
        pass
    def get_distance(self,metric):
        VALID_METRICS=['eucledian','manhattan']
        if metric not in VALID_METRICS:
            raise ValueError(f"Please select a valid distance metric:{VALID_METRICS}")
        if metric == "eucledian":
            return "a"
    def eucledian_distance(self,p1,p2):
        euc_distance=np.sqrt(np.sum((p1-p2)**2))
        if math_breakdown:
            self.steps+=f"\neucledian distance calc between points ({p1,p2}): {euc_distance}"
        return euc_distance
    def manhattan_distance(self,p1,p2):
        man_distance=np.sum(np.abs(p1,p2))
        if math_breakdown:
            self.steps+=f"\nmanhattan distance calc between points ({p1,p2}): {man_distance}"
        return man_distance
    
    

        


if __name__=="__main__":
    print("im alive")
    