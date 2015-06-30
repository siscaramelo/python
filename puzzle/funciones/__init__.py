#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir

'''
from numpy import *
import copy


def mGetTile(vHashValue):
    m = 0
    dimension   = int(sqrt(len(vHashValue)))
    estado      = []
       
    for i in range(dimension):
        estadoTmp   = []
        for j in range(dimension):
            estadoTmp.append(int(vHashValue[m]))
            m += 1 
        estado.append(estadoTmp)    

    return estado

def checkPrecond(n,vParametros,vPrecond,vPrecondiciones):
    for vEstado in vPrecond[n]:
        p = copy.deepcopy(vParametros)
        for i in range(1,len(p)):
            if p[vPrecondiciones[n][i]]==None:
                p[vPrecondiciones[n][i]]=vEstado[i]
                
            
        vTestCondicion = []
        vTestCondicion.append(vPrecondiciones[n][0])
        for i in range(1,len(p)):
            vTestCondicion.append(p[vPrecondiciones[n][i]])

        if vEstado == vTestCondicion:
            n += 1
            if n < len(vPrecond):
                checkPrecond(n,p,vPrecond,vPrecondiciones)
                return vEstado
            else:
                print vEstado
        else:
            return False
        
    
