# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

### Linear Regression Using Gradient Descent Algorithm

1. **Import Libraries**: Load `numpy`, `pandas`, `StandardScaler`.
2. **Define Function**: Create `linear_regression(X1, y, learning_rate, num_iters)`.
3. **Prepare Data**: Add intercept column to X1 and initialize theta.
4. **Gradient Descent**: Update theta for a specified number of iterations.
5. **Load Dataset**: Read CSV file (e.g., `50_startups.csv`).
6. **Extract Features/Target**: Separate X and y, reshape as needed.
7. **Scale Data**: Standardize X1 and y using `StandardScaler`.
8. **Train Model**: Call `linear_regression` with scaled data.
9. **Make Predictions**: Scale new input data and calculate predictions.
10. **Inverse Scale**: Transform predictions back to original scale.
11. **Output Results**: Print predicted values.

### End of Algorithm

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Ashwin Kumar A
RegisterNumber:  212223040021
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate * (1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_startups.csv",header=None)
data.head()
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data= np.array([165349.2 , 136897.8 , 471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_scaled), theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:

![image](https://github.com/user-attachments/assets/5bffe586-d187-4ce6-a387-861d0ba2f635)

![image](https://github.com/user-attachments/assets/07a6b2e5-0dd6-4f74-ae74-2ec7968e88a7)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
