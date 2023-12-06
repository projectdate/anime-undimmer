import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv('regression.csv', header=None)

# Split the data into features (X) and target (y)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]

# Create and train the model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

# Print the parameters
print('Feature importance: ', model.feature_importances_)
from sklearn.metrics import accuracy_score

# Predict the target for the features
y_pred = model.predict(X)

# Print the accuracy of the model
print('Accuracy: ', accuracy_score(y, y_pred))

# Print the points that the model got wrong
print('Points the model got wrong: ', X[y != y_pred])

def predict(x1, x2):
    return model.predict([[x1, x2]])

# Print a sample x and y and prediction
sample_x = X.iloc[0]
sample_y = y.iloc[0]
print('Sample X: ', sample_x)
print('Sample Y: ', sample_y)
print('Prediction for sample X: ', predict(sample_x[0], sample_x[1]))

from sklearn import tree
import matplotlib.pyplot as plt

# Print the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(model, filled=True, ax=ax)
plt.show()

# Print the parameters of the decision tree
print('Parameters of the decision tree: ', model.get_params())
