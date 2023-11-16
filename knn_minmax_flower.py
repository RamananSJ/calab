import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler #for sub (iv)
from sklearn.preprocessing import MinMaxScaler

# Min-Max Standardization
def min_max_scaling(dataset, feature):
    min_val = dataset[feature].min()
    max_val = dataset[feature].max()
    return (dataset[feature] - min_val) / (max_val - min_val)



#caculating distances
def euclid(dataset):
  new_record=dataset.iloc[-1,:]
  dataset=dataset.iloc[:-1,:]#removing last row
  dataset['Distance'] = np.sqrt((dataset['Sepal-length'] - new_record['Sepal-length'])**2 + (dataset['Sepal-width'] - new_record['Sepal-width'])**2 +
                                (dataset['Petal-length'] - new_record['Petal-length'])**2 + (dataset['Petal-width'] - new_record['Petal-width'])**2)

  # Sort the data by distance and select the top k=9 neighbors
  k = 12
  nearest_neighbors = dataset.sort_values(by='Distance')[:k]

  # Count the class occurrences among the nearest neighbors
  class_counts = nearest_neighbors['Ground-truth'].value_counts()

  # Classify the new record using unweighted voting
  predicted = class_counts.idxmax()

  print(f"The predicted class for the new record is: {predicted}")
  print(dataset['Distance'])

def manhat(dataset):
  new_record=dataset.iloc[-1,:]
  dataset=dataset.iloc[:-1,:]#removing last row
  dataset['Distance'] = (abs(dataset['Sepal-length'] - new_record['Sepal-length']) + abs(dataset['Sepal-width'] - new_record['Sepal-width']) +
                                abs(dataset['Petal-length'] - new_record['Petal-length']) + abs(dataset['Petal-width'] - new_record['Petal-width']))

  # Sort the data by distance and select the top k=9 neighbors
  k = 12
  nearest_neighbors = dataset.sort_values(by='Distance')[:k]

  # Count the class occurrences among the nearest neighbors
  class_counts = nearest_neighbors['Ground-truth'].value_counts()

  # Classify the new record using unweighted voting
  predicted = class_counts.idxmax()

  print(f"The predicted class for the new record is: {predicted}")
  print(dataset['Distance'])


dataset = pd.read_csv("data4_19.csv")

dataset['Sepal-length'] = min_max_scaling(dataset, 'Sepal-length')
dataset['Sepal-width'] = min_max_scaling(dataset, 'Sepal-width')
dataset['Petal-length'] = min_max_scaling(dataset, 'Petal-length')
dataset['Petal-width'] = min_max_scaling(dataset, 'Petal-width')

euclid(dataset)
print('\n\n\n')
manhat(dataset)
