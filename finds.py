#with dataset
import pandas as pd
import numpy as np
d = pd.read_csv("dataset.csv") 
print(d)
#dataset should not have headings
a = np.array(d)[:,:-1] 
print(" The attributes are: ",a)
t = np.array(d)[:,-1] 
print("The target is: ",t) 
def fun(c,t): 
 for i, val in enumerate(t): 
 if val == "Yes": 
 specific_hypothesis = c[i].copy() 
 break 
  
 for i, val in enumerate(c): 
 if t[i] == "Yes": 
 for x in range(len(specific_hypothesis)): 
 if val[x] != specific_hypothesis[x]: 
 specific_hypothesis[x] = '?' 
 else: 
 pass  
 return specific_hypothesis 
print(" The final hypothesis is:",fun(a,t))

#without dataset
import numpy as np
import pandas as pd
# Define your own dataset
data = {
    'Attribute1': ['Circular','Circular','Oval','Oval'],
    'Attribute2': ['Large','Large','Large','Large'],
    'Attribute3': ['Light','Light','Dark','Light'],
    'Attribute4': ['Smooth','Irregular','Smooth','Irregular'],
    'Attribute5': ['Thick','Thick','Thin','Thick'],
    'Target': ['Yes', 'Yes', 'No', 'Yes']
}

# Create a pandas DataFrame
d = pd.DataFrame(data)

print("Original dataset:")
print(d)

# Extract attributes and target
a = np.array(d)[:, :-1]
t = np.array(d)[:, -1]

# Define the Find-S algorithm
def find_s_algorithm(c, t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break

    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass

    return specific_hypothesis

# Apply the Find-S algorithm
final_hypothesis = find_s_algorithm(a, t)
print("The final hypothesis is:", final_hypothesis)
