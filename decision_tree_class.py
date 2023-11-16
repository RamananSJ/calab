#Decision tree for sunny windy problem
#Dataset
#     Outlook  Temp Humidity   Windy PlayTennis
#0      Sunny   Hot     High    Weak         No
#1      Sunny   Hot     High  Strong         No
#2   Overcast   Hot     High    Weak        Yes
#3       Rain  Mild     High    Weak        Yes
#4       Rain  Cool   Normal    Weak        Yes
#5       Rain  Cool   Normal  Strong         No
#6   Overcast  Cool   Normal  Strong        Yes
#7      Sunny  Mild     High    Weak         No
#8      Sunny  Cool   Normal    Weak        Yes
#9       Rain  Mild   Normal    Weak        Yes
#10     Sunny  Mild   Normal  Strong        Yes
#11  Overcast  Mild     High  Strong        Yes
#12  Overcast   Hot   Normal    Weak        Yes
#13      Rain  Mild     High  Strong         No

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import graphviz

def importdata(path = './DT.csv'):
    data = pd.read_csv(path, header = 0, skiprows = 0)
    print ("Dataset Length: ", len(data))
    print ("Dataset Shape: ", data. shape)
    #data.target = data['play']
    return data
playtennis_data = importdata()

#creating a Label encoder to convert text to numbers
Le = LabelEncoder()
playtennis_data['Outlook'] = Le.fit_transform(playtennis_data['Outlook'])
playtennis_data['Temp'] = Le.fit_transform(playtennis_data['Temp']) 
playtennis_data['Humidity'] = Le.fit_transform(playtennis_data['Humidity'])  
playtennis_data['Windy'] = Le.fit_transform(playtennis_data['Windy']) 
playtennis_data['PlayTennis'] = Le.fit_transform(playtennis_data['PlayTennis']) 
X = playtennis_data[['Outlook', 'Temp', 'Humidity', 'Windy']]
y = playtennis_data['PlayTennis']
X = playtennis_data[['Outlook', 'Temp', 'Humidity', 'Windy']]
y = playtennis_data['PlayTennis']
# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy', random_state=0) 
tree.fit(X,y)
dot_data = export_graphviz(tree, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("iris")
dot_data = export_graphviz(tree, out_file=None, feature_names=
['Outlook', 'Temp', 'Windy', 'Humidity'], class_names= ['No', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source (dot_data)
graph
