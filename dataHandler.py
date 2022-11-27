import pandas as pd
import numpy as np 
import glob
import os

#edge cases
#consecutive GSO
#consecutive ball starts

def getStartSide(indexOfPlayStart):
    sside=0
    match df['Brief'][indexOfPlayStart][4:]:
        case 'BS1':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=11
                case 'Centr':
                    sside=12
                case 'Right':
                    sside=13
                case _:
                    sside=444                                           #error in finding start side
        case 'BS2':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=21
                case 'Centr':
                    sside=22
                case 'Right':
                    sside=23
                case _:
                    sside=444
        case 'BS3':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=31
                case 'Centr':
                    sside=32
                case 'Right':
                    sside=33
                case _:
                    sside=444
        case 'BS4':
            match df['Detailed'][indexOfPlayStart+1]*5:
                case 'Left ':
                    sside=41
                case 'Centr':
                    sside=42
                case 'Right':
                    sside=43
                case _:
                    sside=444

        case 'TO1':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=43
                case 'Centr':
                    sside=42
                case 'Right':
                    sside=41
                case _:
                    sside=444
        case 'TO2':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=33
                case 'Centr':
                    sside=32
                case 'Right':
                    sside=31
                case _:
                    sside=444
        case 'TO3':
            match df['Detailed'][indexOfPlayStart+1]*5:
                case 'Left ':
                    sside=23
                case 'Centr':
                    sside=22
                case 'Right':
                    sside=21
                case _:
                    sside=444
        case 'TO4':
            match df['Detailed'][indexOfPlayStart+1][:5]:
                case 'Left ':
                    sside=13
                case 'Centr':
                    sside=12
                case 'Right':
                    sside=11
                case _:
                    sside=444
    
    return sside
        
def getEndSide(i):

    eside=0
    match df['Brief'][i-1]:
        case 'Own PCA' | 'Own GSO':            
            match df['Detailed'][i-3]*5:        
                case 'Left ':
                    eside=41
                case 'Centr':
                    eside=42
                case 'Right':
                    eside=43
                case _:
                    eside=42        #this would be where a PCA leads to another PCA or to a GSO
        case _:                     #this would be the default case
            match df['Detailed'][i-1][:5]:  
                case 'Left ':
                    eside=41
                case 'Centr':
                    eside=42
                case 'Right':
                    eside=43
                case _:
                    eside=444                                       #error in finding end side

    return eside

outdf = pd.DataFrame(columns=['Start&Side','Finish&Side','GSO'])

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
        #print('miss '+str(i))
        startSide = getStartSide(indexOfPlayStart)
        endSide = getEndSide(i)
        tmpdf = pd.DataFrame(np.array([startSide,endSide,-1]).reshape(1,-1), columns=['Start&Side','Finish&Side','GSO'])
        outdf = pd.concat([outdf,tmpdf],ignore_index=True)   

    elif(df['Brief'][i] == 'Own GSO'):                                                                                                              #positive attack            account for consecutive GSO here
        hit+=1                      #for testing
        potentialGSO=False
        #print('hit'+str(i))
        startSide = getStartSide(indexOfPlayStart)
        endSide = getEndSide(i)
        tmpdf = pd.DataFrame(np.array([startSide,endSide,1]).reshape(1,-1), columns=['Start&Side','Finish&Side','GSO'])
        outdf = pd.concat([outdf,tmpdf],ignore_index=True)   
    
outdf.to_csv('hockey.csv', index=False)
    

print("hits: "+ str(hit))
print("fails: "+ str(fail))
 



