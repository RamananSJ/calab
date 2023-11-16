import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import graphviz

def importdata(path = './DecisionTree.csv'):
    data = pd.read_csv(path, header = 0, skiprows = 0)
    print(data)
    print ("Dataset Length: ", len(data))
    print ("Dataset Shape: ", data. shape)
    #data.target = data['play']
    return data
playtennis_data = importdata()

#creating a Label encoder to convert text to numbers
Le = LabelEncoder()
playtennis_data['DEPT'] = Le.fit_transform(playtennis_data['DEPT'])
playtennis_data['COUNT'] = Le.fit_transform(playtennis_data['COUNT']) 
playtennis_data['AGE'] = Le.fit_transform(playtennis_data['AGE'])  
playtennis_data['SALARY'] = Le.fit_transform(playtennis_data['SALARY']) 
playtennis_data['STATUS'] = Le.fit_transform(playtennis_data['STATUS']) 
X = playtennis_data[['DEPT','COUNT','AGE','SALARY']]
y = playtennis_data['STATUS']

# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy', random_state=0) 
tree.fit(X,y)
dot_data = export_graphviz(tree, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("iris")
dot_data = export_graphviz(tree, out_file=None, feature_names=
['DEPT','COUNT','AGE','SALARY'], class_names= ['Junior', 'Senior'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source (dot_data)
graph
