import pandas as pd
import numpy as np 
import glob
import os

#edge cases
#consecutive GSO
#consecutive ball starts

outdf = pd.DataFrame(columns=['Start&Side','Finish&Side','GSO'])
print(outdf)

cwd = os.getcwd()
path=os.path.join(cwd,'data\individual files')
all_files = glob.glob(os.path.join(path , "*.csv"))

hit=0       #testing
fail=0      #testing

#for filename in all_files:
filename=str(path)+'\Game 1.csv'
df = pd.read_csv(filename, header=1)                                                #need to remove header or run into length mismatch error with column names 
columns = ['Start','End','Category','Descriptors','Brief','Detailed']            #named column 5 'Brief' and column 6 'Detailed'
df.columns = columns

potentialGSO=False
indexOfPlayStart=0

for i in df.index:
    if((df['Brief'][i] == 'Own BS1') | (df['Brief'][i] == 'Own BS2') | (df['Brief'][i] == 'Own BS3') | (df['Brief'][i] == 'Own BS4')):              #Own ballstarts             account for consecutuve ball start here
        potentialGSO=True
        indexOfPlayStart=i
    if((df['Brief'][i] == 'Opp TO1') | (df['Brief'][i] == 'Opp TO2') | (df['Brief'][i] == 'Opp TO3') | (df['Brief'][i] == 'Opp TO4')):              #Opp turnovers
        potentialGSO=True
        indexOfPlayStart=i

    if(((df['Brief'][i] == 'Own TO4') | (df['Brief'][i] == 'Own TO C')) & potentialGSO==True):                                                      #failed attack
        fail+=1                     #for testing
        potentialGSO=False
        print(indexOfPlayStart)


    elif(df['Brief'][i] == 'Own GSO'):                                                                                                              #positive attack            account for consecutive GSO here
        hit+=1                      #for testing
        potentialGSO=False


    

print("hits: "+ str(hit))
print("fails: "+ str(fail))

