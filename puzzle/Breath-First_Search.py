'''
Created on 06/02/2015

@author: pgiraldez
'''

# Importamos las funciones y clases necesarias
from clases import *
from funciones import *
from numpy import *
import copy


# Empezamos el programa principal

if __name__ == '__main__':
    
    # Variables iniciales
    lSucesores      = cListaSucesores()
    lCostesEstados  = cListaCosteTarea()
    
    nivel           = 0
    Depth           = 27
    sucesor         = []
    colaDeTrabajo   = []

    Resuelto = False
        
    # Cargamos el estado inicial
    puzzle          = cPuzzle([[0,1,2],[3,4,5],[6,7,8]])
    
    # Imprimimos el punto de partida del puzzle
    print 'Estado Inicial:\n'
    print puzzle
    
    print 'Columnas: ', puzzle.getColumnas()
    print 'Filas:    ', puzzle.getLineas()
    print 'Posicion del cero: ', puzzle.getPosicionCero()
    
    # Cargamos el estado de partida
    lCostesEstados.addEstado(puzzle.hashValue, nivel)
    colaDeTrabajo.append(puzzle.hashValue)
    
    while len(colaDeTrabajo) > 0 and nivel<Depth+1 and Resuelto==False:
        #nivel += 1
        
        proxEstado = colaDeTrabajo[0]
        lSucesores.addEstado(proxEstado,lSucesores.obtenerPadre(proxEstado))
        lProxEstado = cPuzzle(mGetTile(proxEstado))
        nivel = lCostesEstados.getNivelEstado(proxEstado)+1
       
        for i in range(4):
            
            sucesor = copy.deepcopy(lProxEstado)
            
            if sucesor.moveTile(i):
                
                # Comprobamos si el nuevo estado esta en la lista
                if not lSucesores.comprobarLista(sucesor.hashValue):
                    
                    # Incluimos al sucesor y a su padre en una lista
                    lSucesores.addEstado(sucesor.hashValue, proxEstado)
                    lCostesEstados.addEstado(sucesor.hashValue, nivel)
                    colaDeTrabajo.append(sucesor.hashValue)
                    

            sucesor=[]
            
        colaDeTrabajo.remove(proxEstado)
            
    print 'Numero de sucesores',len(lSucesores)
    print 'Numero de estados nivel: ', Depth,lCostesEstados.noEstadosNivel(Depth)
    print 'Valor maximo de la cola de Trabajo: ', max(colaDeTrabajo)
    
    