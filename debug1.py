import pandas as pd
dataset1 = pd.read_csv('double_deep_learning.csv')
dataset2 = pd.read_csv('submission_final_deeep3.csv')
    #dataset3 = pd.read_csv('test.csv')
y1 = dataset1.iloc[:, 1].values
y3 = dataset1.iloc[:, 0].values
y2 = dataset2.iloc[:, 1].values
    
    # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y1, y2)
    
q=0
for z in range(425):
    if y1[z] != y2[z]:
        print(y1[z]," ", y2[z], "index:", y3[z])
        q += 1
            
print((500-q)/500.0, "   ", q)