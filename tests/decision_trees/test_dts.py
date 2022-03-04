import unittest 
import sys
sys.path.insert(1,"/home/danieldroder/Coding/machine_learning_explained/ml_from_scratch/supervised/")
from decision_trees import decision_tree
# class testDecisionTrees(unittest.TestCase)
class test(unittest.TestCase):
    def test_initialize(self):
        model=decision_tree()
        print(f"{model.min_samples}")

# if __name__=="__main__":
#     tests=test()
    