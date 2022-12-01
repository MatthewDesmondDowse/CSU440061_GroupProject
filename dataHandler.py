import pandas as pd
import numpy as np 
import glob
import os
from getSides import getStartSide,getEndSide

#edge cases
#consecutive GSO
#consecutive ball starts
#hb att
#TO
#breaches
#only left with no other chars(only take slice of 4)
#may need to remove some rows with 0 in any cell(no side)

outdf = pd.DataFrame(columns=['Start&Side','Finish&Side','GSO'])

cwd = os.getcwd()
path=os.path.join(cwd,'data')
all_files = glob.glob(os.path.join(path , "*.csv"))

hit=0       #testing
fail=0      #testing

for filename in all_files:
#filename=str(path)+'\Game 13.csv'    #testing
    print(str(fail+hit+1)+' '+filename)  #testing
    df = pd.read_csv(filename, header=1)                                                #need to remove header or run into length mismatch error with column names 
    columns = ['Start','End','Category','Descriptors','Brief','Detailed']            #named column 5 'Brief' and column 6 'Detailed'
    df.columns = columns

    potentialGSO=False
    indexOfPlayStart=0
    startSide=0
    endSide=0

    for i in df.index:
        if((df['Brief'][i] == 'Own BS1') | (df['Brief'][i] == 'Own BS2') | (df['Brief'][i] == 'Own BS3') | (df['Brief'][i] == 'Own BS4')):              #Own ballstarts             think consecutive ballstarts are handle by indexOFplayStart being updated for each new ballstart
            potentialGSO=True
            indexOfPlayStart=i
        if((df['Brief'][i] == 'Opp TO1') | (df['Brief'][i] == 'Opp TO2') | (df['Brief'][i] == 'Opp TO3') | (df['Brief'][i] == 'Opp TO4')):              #Opp turnovers
            potentialGSO=True
            indexOfPlayStart=i

        if(((df['Brief'][i] == 'Own TO4') | (df['Brief'][i] == 'Own TO C')) & potentialGSO==True):                                                      #failed attack
            fail+=1                     #for testing
            potentialGSO=False
            #print(str(fail+hit+1) + ' ind '+str(indexOfPlayStart)+' miss '+str(i))  #testing
            startSide = getStartSide(indexOfPlayStart, df)
            endSide = getEndSide(i, df)
            tmpdf = pd.DataFrame(np.array([startSide,endSide,-1]).reshape(1,-1), columns=['Start&Side','Finish&Side','GSO'])
            outdf = pd.concat([outdf,tmpdf],ignore_index=True)   

        elif(df['Brief'][i] == 'Own GSO'):                                                                                                              #positive attack            account for consecutive GSO here
            hit+=1                      #for testing
            potentialGSO=False
            #print(str(hit+fail+1)+' ind '+str(indexOfPlayStart)+' hit '+str(i))     #testing
            startSide = getStartSide(indexOfPlayStart, df)
            endSide = getEndSide(i, df)
            tmpdf = pd.DataFrame(np.array([startSide,endSide,1]).reshape(1,-1), columns=['Start&Side','Finish&Side','GSO'])
            outdf = pd.concat([outdf,tmpdf],ignore_index=True)   
        
outdf.to_csv('hockey.csv', index=False)
os.startfile('hockey.csv')              #testing
    

print("hits: "+ str(hit))
print("fails: "+ str(fail))
 



