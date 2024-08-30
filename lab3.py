import numpy as np
import pandas as pd

data=pd.read_csv(r"C:\test\ML\ENJOYSPORT.csv")

concepts=np.array(data.iloc[:,0:-1])
target=np.array(data.iloc[:,-1])
print("target: {}".format(target))

def learn(concepts, target):
    specific_h=concepts[0].copy()
    print("Intiation of specific_H\n",specific_h)
    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("Intitalization of general_h\n", general_h)

    for i,h in enumerate(concepts):
        if target[i]==1:
            print("Instance is positive")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    specific_h[x]="?"
                    general_h[x][x]="?"
        
        if target[i]==0:
            print("Instance is negative")
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]="?"
        print("step {}".format(i+1))
        print(specific_h)
        print(general_h)
        print("\n\n")

    general_h = [val for val in general_h if val != ['?', '?', '?', '?', '?', '?']]
    print("Final result")
    print("Specific hypothesis :-",specific_h)
    print("General hypothesis :-",general_h)
    
    return specific_h, general_h

s_final, g_final=learn(concepts, target)