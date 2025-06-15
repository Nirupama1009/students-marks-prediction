import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4.5, 5, 5.5, 6.5, 7, 8, 9],
    'Marks': [20, 35, 50, 55, 65, 68, 75, 78, 88, 95]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Hours']]  # input
y = df['Marks']    # output

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 7.5 hours of study
hours = 7.5
predicted = model.predict([[hours]])
print(f"Predicted score for {hours} hours: {predicted[0]:.2f}")

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Hours vs Marks')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.show()