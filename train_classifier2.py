import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model2 = RandomForestClassifier()
model2.fit(x_train, y_train)

# Predict on test data
y_predict = model2.predict(x_test)

# Evaluate with accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Additional evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predict))

print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Save the model
with open('model2.p', 'wb') as f:
    pickle.dump({'model': model2}, f)
