# Description
This package is designed for learning the mathematics behind machine learning algorithms. 

While building the algorithms, it will give step by step instructions as to what is going on. 


# Installation 
First, we must install the library. This can be done by running:
```
pip3 install ml-from-scratch
```


# Usage
```code Python
c=linear_regression(help=True,math_breakdown=True)
c.simple_dataset()
c.train()
print(c.predict([[1,3]]))   
print(c.steps)
```

We can toggle Help and math_breakdown to tell us information about the algorithm we are using (linear regression in this case). And we can use the math_breakdown feature to add some verbosity to the output of what is going on mathematically.


