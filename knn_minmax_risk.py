import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler #for sub (iv)
from sklearn.preprocessing import MinMaxScaler

# Min-Max Standardization
def min_max_scaling(dataset, feature):
    min_val = dataset[feature].min()
    max_val = dataset[feature].max()
    print(min_val)
    print(max_val)
    return (dataset[feature] - min_val) / (max_val - min_val)

def minmax(dataset):
  print(dataset)
  dataset['Age'] = min_max_scaling(dataset, 'Age')
  dataset['Income'] = min_max_scaling(dataset, 'Income')
  print(dataset)

  new_record=dataset.iloc[-1,:]
  dataset=dataset.iloc[:-1,:]#removing row 10

  #caculating distances
  dataset['Distance'] = np.sqrt((dataset['Age'] - new_record['Age'])**2 + (dataset['Income'] - new_record['Income'])**2)

  # Sort the data by distance and select the top k=9 neighbors
  k = 9
  nearest_neighbors = dataset.sort_values(by='Distance')[:k]

  # Count the class occurrences among the nearest neighbors
  class_counts = nearest_neighbors['Risk'].value_counts()

  # Classify the new record using unweighted voting
  predicted_risk_factor = class_counts.idxmax()

  print(f"The predicted Risk Factor for the new record is: {predicted_risk_factor}")
  print(dataset)


def stand(data):
  # Min-Max Standardization
  mm = StandardScaler()

  data.iloc[:,[1,3]] = mm.fit_transform(data.iloc[:,[1,3]])

  new_record = data.iloc[-1,:]
  data = data.iloc[:-1,:]

  # New Record (Record #10) - You can replace these values with your own data

  # Calculate distances
  data['Distance'] = abs(data['Age'] - new_record['Age']) + abs(data['Income'] - new_record['Income']) #Manhattan Distance

  # Sort the data by distance and select the top k=9 neighbors
  k = 9
  nearest_neighbors = data.sort_values(by='Distance')[:k]

  # Count the class occurrences among the nearest neighbors
  class_counts = nearest_neighbors['Risk'].value_counts()

  # Classify the new record using unweighted voting
  predicted_risk_factor = class_counts.idxmax()

  print(f"The predicted Risk Factor for the new record is: {predicted_risk_factor}")
  print(data)


file = pd.read_csv("Data.csv")
file['Income'] = file['Income'].str.replace('[\$,]', '', regex=True).astype(float)
minmax(file)
print('\n\n')
stand(file)
