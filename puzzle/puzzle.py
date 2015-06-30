'''
Created on 01/02/2015

@author: pablo
'''

from clases import *
from funciones import *
from numpy import *

import copy

#from h5py._hl.dims import DimensionManager

#===============================================================================
# def mGetTile(vHashValue):
#     m = 0
#     dimension   = int(sqrt(len(vHashValue)))
#     estado      = []
#        
#     for i in range(dimension):
#         estadoTmp   = []
#         for j in range(dimension):
#             estadoTmp.append(int(vHashValue[m]))
#             m += 1 
#         estado.append(estadoTmp)    
# 
#     return estado
#===============================================================================

if __name__ == '__main__':
    
    # Cargamos el estado inicial
    # puzzle          = cPuzzle([[1,6,4],[8,7,0],[3,2,5]])
    puzzle          = cPuzzle([[8,1,7],[4,5,6],[2,0,3]])
    goal            = cPuzzle([[0,1,2],[3,4,5],[6,7,8]])
    lSucesores      = cListaSucesores()
    lCostesEstados  = cListaCosteTarea()
    
    nivel       = 0
    minCoste    = None
    
    sucesor     = []
    minEstado   = []
    
    Resuelto = False
    
    #===========================================================================
    # print puzzle.estado==goal.estado
    # print puzzle.comparaLista(goal.estado,nivel)
    #===========================================================================
    
    # Imprimimos el punto de partida del puzzle
    print 'Estado Inicial:\n'
    print puzzle
    
    print 'Columnas: ', puzzle.getColumnas()
    print 'Filas:    ', puzzle.getLineas()
    print 'Posicion del cero: ', puzzle.getPosicionCero()
    
    print '\nImprimimos los sucesores:\n'
    
    # Cargamos el estado de partida
    lCostesEstados.addEstado(puzzle.hashValue, goal.comparaLista(puzzle.estado,nivel))
    
#    print lCostesEstados.listaCostes
    
    while len(lCostesEstados) > 0 and nivel<200000 and Resuelto==False:
        nivel += 1
        # print '\n\nNivel: ', nivel
        # print 'Prox. Estado: ', lCostesEstados.proxEstado()
        # print '------------------------------\n'
        
        proxEstado = lCostesEstados.proxEstado()
        lSucesores.addEstado(proxEstado,lSucesores.obtenerPadre(proxEstado))
        lProxEstado = cPuzzle(mGetTile(proxEstado))
       
        for i in range(4):
            
            sucesor = copy.deepcopy(lProxEstado)
            
            if sucesor.moveTile(i):
                
                # Comprobamos si el nuevo estado esta en la lista
                if not lSucesores.comprobarLista(sucesor.hashValue):
                    
                    # Incluimos al sucesor y a su padre en una lista
                    lSucesores.addEstado(sucesor.hashValue, proxEstado)
                    lCostesEstados.addEstado(sucesor.hashValue, goal.comparaLista(sucesor.estado,nivel))
                    
                    #print sucesor[i]
                    
                    # Comprobamos si hemos encontrado la solucion
                    if goal.estado==sucesor.estado:
                        # Si encontramos el estado final
                        Resuelto = True
                        print 'Resuelto el puzzle'
                    else:
                        #lCostesEstados.delEstado(lProxEstado.hashValue)
                        vCoste = goal.comparaLista(sucesor.estado,nivel)
                        if minCoste == None or minCoste > vCoste-nivel:
                            minCoste  = vCoste-nivel
                            minEstado = sucesor.hashValue
                            
                        # print 'Coste: ', vCoste-nivel 
                        # print

            sucesor=[]
        # print lSucesores
        lCostesEstados.delEstado(proxEstado)
    
    print '\n\nLista de sucesores:'
    print '-------------------\n'
    
    if (Resuelto):
        lSucesores.imprimirPath(goal.hashValue)
    else:
        print 'Puzzle NO RESUELTO!!!!'
        print 'Estado minimo: ', minEstado, minCoste
        
    print 'Sucesores examinados:', nivel    
