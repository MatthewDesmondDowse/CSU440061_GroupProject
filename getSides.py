def getStartSide(indexOfPlayStart, df):
    sside=0
    match df['Brief'][indexOfPlayStart][4:]:
        case 'TO1':
            match str(df['Detailed'][indexOfPlayStart])[:5]:
                case 'Left ':
                    sside=43
                case 'Centr' | 'BLine' | 'Bline':
                    sside=42
                case 'Right':
                    sside=41
                case 'TO' | 'Breac':
                    sside=777
                case '75 Pr':
                    sside=666
                case _:
                    sside=444  
        case 'TO2':
            match str(df['Detailed'][indexOfPlayStart])[:5]:
                case 'Left ':
                    sside=33
                case 'Centr' | 'BLine' | 'Bline':
                    sside=32
                case 'Right':
                    sside=31
                case 'TO' | 'Breac':
                    sside=777
                case '75 Pr':
                    sside=666
                case _:
                    sside=444 
        case 'TO3':
            match str(df['Detailed'][indexOfPlayStart])[:5]:
                case 'Left ':
                    sside=23
                case 'Centr' | 'BLine' | 'Bline':
                    sside=22
                case 'Right':
                    sside=21
                case 'TO' | 'Breac':
                    sside=777
                case '75 Pr':
                    sside=666
                case _:
                    sside=444 
        case 'TO4':
            match str(df['Detailed'][indexOfPlayStart])[:5]:
                case 'Left ':
                    sside=13
                case 'Centr' | 'BLine' | 'Bline':
                    sside=12
                case 'Right':
                    sside=11
                case 'TO' | 'Breac':
                    sside=777
                case _:
                    sside=444 

        case 'BS1':
            match str(df['Detailed'][indexOfPlayStart+1])[:5]:
                case 'Left ':
                    sside=11
                case 'Centr' | 'BLine' | 'Bline':
                    sside=12
                case 'Right':
                    sside=13
                case 'TO' | 'Breac':
                    match str(df['Detailed'][indexOfPlayStart+3])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=555
                case '75 Pr':
                    match str(df['Detailed'][indexOfPlayStart+2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=888
                case 'CP':                                                      #if bs is cp then must be centre
                    sside=42
                case _:
                    match str(df['Detailed'][indexOfPlayStart-2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=444                                            #error in finding start side
        case 'BS2':
            match str(df['Detailed'][indexOfPlayStart+1])[:5]:
                case 'Left ':
                    sside=21
                case 'Centr' | 'BLine' | 'Bline':
                    sside=22
                case 'Right':
                    sside=23
                case 'TO' | 'Breac':
                    match str(df['Detailed'][indexOfPlayStart+3])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=555
                case '75 Pr':
                    match str(df['Detailed'][indexOfPlayStart+2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=888
                case 'CP':
                    sside=42
                case _:
                    match str(df['Detailed'][indexOfPlayStart-2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=444 
        case 'BS3':
            match str(df['Detailed'][indexOfPlayStart+1])[:5]:
                case 'Left ':
                    sside=31
                case 'Centr' | 'BLine' | 'Bline':
                    sside=32
                case 'Right':
                    sside=33
                case 'TO' | 'Breac':
                    match str(df['Detailed'][indexOfPlayStart+3])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=555
                case '75 Pr':
                    match str(df['Detailed'][indexOfPlayStart+2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=888
                case 'CP':
                    sside=42
                case _:
                    match str(df['Detailed'][indexOfPlayStart-2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=444 
        case 'BS4':
            match str(df['Detailed'][indexOfPlayStart+1])[:5]:
                case 'Left ':
                    sside=41
                case 'Centr' | 'BLine' | 'Bline':
                    sside=42
                case 'Right':
                    sside=43
                case 'TO' | 'Breac':
                    match str(df['Detailed'][indexOfPlayStart+3])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=555
                case '75 Pr':
                    match str(df['Detailed'][indexOfPlayStart+2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=888
                case 'CP':
                    sside=42
                case _:
                    match str(df['Detailed'][indexOfPlayStart-2])[:5]:
                        case 'Left ':
                            sside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            sside=42
                        case 'Right':
                            sside=43
                        case 'TO' | 'Breac':
                            sside=777
                        case '75 Pr':
                            sside=666
                        case _:
                            sside=444
    
    return sside
        
def getEndSide(i, df):

    eside=0
    match str(df['Brief'][i-1])[:6]:
        case 'Own PC' | 'Own GS' | 'Opp Sh' | 'Own Sh':                    #not sure about shuttle, putting it here for the moment
            match str(df['Detailed'][i-3])[:5]:        
                case 'Left ':
                    eside=41
                case 'Centr' | 'BLine' | 'Bline':
                    eside=42
                case 'Right':
                    eside=43
                case _:
                    eside=42        #this would be where a PCA leads to another PCA or to a GSO     
        case _:                     #this would be the default case
            match str(df['Detailed'][i-1])[:5]:  
                case 'Left ':
                    eside=41
                case 'Centr' | 'BLine' | 'Bline':
                    eside=42
                case 'Right':
                    eside=43
                case _:
                    match str(df['Detailed'][i])[:5]:  
                        case 'Left ':
                            eside=41
                        case 'Centr' | 'BLine' | 'Bline':
                            eside=42
                        case 'Right':
                            eside=43
                        case _:
                            match str(df['Brief'][i])[:6]:
                                case 'Own TO':
                                    match str(df['Detailed'][i])[:4]:  
                                        case 'Left':
                                            eside=41
                                        case 'Cent' | 'BLin':
                                            eside=42
                                        case 'Righ':
                                            eside=43
                                        case 'BS 1' | 'BS 2' | 'BS 3' | 'BS 4':
                                            match (str(df['Detailed'][i])[7:11]):
                                                case 'Left':
                                                    eside=41
                                                case 'Cent' | 'BLin':
                                                    eside=42
                                                case 'Righ':
                                                    eside=43
                                                case _:
                                                    match str(df['Detailed'][i+1])[:4]:  
                                                        case 'Left':
                                                            eside=41
                                                        case 'Cent' | 'BLin':
                                                            eside=42
                                                        case 'Righ':
                                                            eside=43
                                                        case '75 P':
                                                            match str(df['Detailed'][i-1])[:4]:  
                                                                case 'Left':
                                                                    eside=41
                                                                case 'Cent' | 'BLin' | 'Blin':
                                                                    eside=42
                                                                case 'Righ':
                                                                    eside=43
                                                                case _:
                                                                    eside=999
    return eside