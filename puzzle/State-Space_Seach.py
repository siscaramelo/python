'''
Created on 08/02/2015

@author: pablo
'''
import funciones as f
import copy

if __name__ == '__main__':
    
    vInit                =[['S','B','B'],['S','C','B'],['S','A','C'],['R','B','B'],['R','C','B']]
    vParametros          = {'x2': None, 'x3': None, 'x1': None}
    # Action op1: Precondiciones
    vPrecondicionesAND   =[['S','x1','x2'],['R','x3','x1']]
    
    # Action op2: Precondiciones
    #vPrecondicionesAND   =[['S','x3','x1'],['R','x2','x2']]

    print vParametros
    print vPrecondicionesAND
    print vInit
    
    vPrecondAND = []
    
    # Recorremos las precondiciones sobre los estados iniciales
    for i in range(len(vPrecondicionesAND)):
        vPrecondAND.append([])
        for j in range(len(vInit)): 
            if vInit[j][0]==vPrecondicionesAND[i][0]:
                vPrecondAND[i].append(vInit[j])
                
                
    print vPrecondAND            
    
    f.checkPrecond(0,vParametros,vPrecondAND,vPrecondicionesAND)
    
#===============================================================================
#     for i in vPrecondAND[0]:
#         p = copy.deepcopy(vParametros)
#         p[vPrecondicionesAND[0][1]]=i[1]
#         p[vPrecondicionesAND[0][2]]=i[2]
# 
#         for j in vPrecondAND[1]:
#             p2 = copy.deepcopy(p)
#             if p2[vPrecondicionesAND[1][1]]==None:
#                 p2[vPrecondicionesAND[1][1]]=j[1]
#             if p2[vPrecondicionesAND[1][2]]==None:
#                 p2[vPrecondicionesAND[1][2]]=j[2]
#             
#             vTestCondicion = []
#             vTestCondicion.append(vPrecondicionesAND[1][0])
#             vTestCondicion.append(p2[vPrecondicionesAND[1][1]])
#             vTestCondicion.append(p2[vPrecondicionesAND[1][2]])
#             
#             if j==vTestCondicion:
#                 print 'Accion aplicable',i,j
#                 #break
#             else:
#                 print 'Accion NO aplicable',i,j
#===============================================================================
    
    