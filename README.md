

# 🚀 Logistic Regression from Scratch: Implementation and Optimization

This project presents a step-by-step implementation of **logistic regression from scratch**, inspired by a practical homework assignment. It covers:  

✅ **Data preparation**  
✅ **Logistic regression model formulation**  
✅ **Different optimization techniques** (Gradient Descent, SGD, Mini-Batch GD)  
✅ **Impact of L2 regularization**  
✅ **Hyperparameter tuning**  

---

## 📌 Introduction: Understanding Logistic Regression  

Logistic Regression is a fundamental classification algorithm used for **binary outcomes**, predicting the **probability** of an instance belonging to a particular class.  

🔹 Instead of using libraries like `sklearn` for training, this guide walks through **manually implementing logistic regression** using Python and NumPy.  
🔹 Various **optimization algorithms** are explored to minimize the objective function effectively.  
🔹 **Regularization** is applied to prevent overfitting and improve generalization.  

---

## 📊 I. Data Preparation  

Before training, the dataset is **loaded, examined, cleaned, split, and scaled** for better model performance.  

### 1️⃣ Loading and Examining the Data  

We use **Pandas** to load the **Wisconsin Diagnostic Breast Cancer Dataset**:

```python
import pandas as pd
df = pd.read_csv('/data.csv')
df.head()
```

📝 **Preprocessing Steps**:
- **Drop unnecessary columns** (`'id'`, `'Unnamed: 32'`)
- **Convert categorical labels** ('M' → `0`, 'B' → `1`)
- **Check for missing values**

```python
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
```

### 2️⃣ Splitting into Training and Testing Sets  

We split the dataset into **80% training** and **20% testing**:

```python
from sklearn.model_selection import train_test_split
y = df['diagnosis'].values
x = df.drop(['diagnosis'], axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 3️⃣ Feature Scaling (Standardization)  

To ensure numerical stability, features are **standardized**:

```python
import numpy as np
d = x_train.shape[1]
mu = np.mean(x_train, axis=0).reshape(1, d)
sig = np.std(x_train, axis=0).reshape(1, d)
x_train = (x_train - mu) / (sig + 1E-6)
x_test = (x_test - mu) / (sig + 1E-6)
```

---

## 📈 II. The Logistic Regression Model  

### 🔹 The Sigmoid Function  

The sigmoid function is defined as 

![Sigmoid Equation](https://latex.codecogs.com/svg.image?\sigma(z)=\frac{1}{1&plus;e^{-z}})


### 🔹 Objective Function with L2 Regularization  

The **objective function** (negative log-likelihood + L2 regularization):

![Objective Equations](https://latex.codecogs.com/svg.image?&space;Q(w;X,y)=\frac{1}{n}\sum_{i=1}^{n}\log(1&plus;\exp(-y_i&space;x_i^T&space;w))&plus;\frac{\lambda}{2}|w|_2^2&space;)

🔸 First term: **Logistic loss** (measuring prediction error)  
🔸 Second term: **L2 penalty** (reducing model complexity)  

Implementation:

```python
def objective(w, x, y, lam):
    w = w.reshape(-1, 1)
    y = y.reshape(-1, 1)
    logistic_loss = np.mean(np.log(1 + np.exp(-y * (np.dot(x, w)))))
    reg_loss = (lam / 2) * np.linalg.norm(w)**2
    return logistic_loss + reg_loss
```

---

## 🔍 III. Optimization Techniques  

### 🚀 1. Batch Gradient Descent (GD)  

**Batch Gradient Descent** updates the weights using the entire dataset:  

![Batch gradient](https://latex.codecogs.com/svg.image?\nabla&space;Q(w)=-\frac{1}{n}\sum_{i=1}^n\frac{y_i&space;x_i}{1&plus;\exp(y_i&space;x_i^T&space;w)}&plus;\lambda&space;w&space;)

```python
def gradient(w, x, y, lam):
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)
    logistic_loss = -np.mean((y * x) / (1 + np.exp(y * np.dot(x, w))), axis=0).reshape(-1,1)
    reg_loss = w * lam
    return logistic_loss + reg_loss
```

```python
def gradient_descent(x, y, lam, learning_rate, w, max_epoch=100):
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    objvals = []
    for epoch in range(max_epoch):
        grad = gradient(w, x, y, lam)
        obj = objective(w, x, y, lam)
        w -= learning_rate * grad
        objvals.append(obj)
    return w, objvals
```

---

### 🎲 2. Stochastic Gradient Descent (SGD)  

Instead of using the full dataset, **SGD** updates weights **one sample at a time**:  

```python
def sgd(x, y, lam, learning_rate, w, max_epoch=100):
    n, d = x.shape
    objvals = []
    for epoch in range(max_epoch):
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        epoch_objval = 0
        for i in range(n):
            xi = x_shuffled[i].reshape(-1, 1)
            yi = y_shuffled[i]
            q_i, g = stochastic_objective_gradient(w, xi, yi, lam)
            w -= learning_rate * g
            epoch_objval += q_i
        objvals.append(epoch_objval / n)
    return w, objvals
```

---

### 📦 3. Mini-Batch Gradient Descent (MBGD)  

**MBGD** updates weights using small batches of data:

```python
def mbgd(x, y, lam, learning_rate, w, max_epoch=100, batch_size=32):
    n, d = x.shape
    objvals = []
    for epoch in range(max_epoch):
        indices = np.random.permutation(n)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        for i in range(0, n, batch_size):
            xi = x_shuffled[i:i+batch_size, :]
            yi = y_shuffled[i:i+batch_size, :]
            obj, g = mb_objective_gradient(w, xi, yi, lam)
            w -= learning_rate * g
        full_obj, _ = mb_objective_gradient(w, x, y, lam)
        objvals.append(full_obj)
    return w, objvals
```

---

## 📌 IV. Predictions and Model Evaluation  

### 🔹 Making Predictions  

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(w, X):
    y_pred = np.dot(X, w)
    y_pred = sigmoid(y_pred)
    return np.where(y_pred >= 0.5, 1, 0)
```

### 🔹 Comparing Optimization Methods  

We analyze:  
📌 **Convergence speed**  
📌 **Final loss values**  
📌 **Effect of L2 regularization**  

---

## ⚡ VI. Conclusion  

🎯 **Key Takeaways:**  
✅ Implementing logistic regression from scratch helps build intuition about optimization techniques.  
✅ **Regularization** prevents overfitting and improves generalization.  
✅ **Hyperparameter tuning** is crucial for optimal model performance.  

🚀 **Next Steps:** Extend the model to **multi-class classification** using Softmax Regression!  
