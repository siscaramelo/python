#!/usr/bin/env python
'''
Created on 08/01/2014

@author: pabgir

'''
from numpy import *
from matplotlib.cbook import Null

class cPuzzle:
    
    def __init__(self, estadoInicial):
        # Inicializa el puzzle
        self.lineas, self.columnas = shape(estadoInicial)
        self.estado = estadoInicial
        self.hashValue = self.mhashValue()
 #       self.posicionCero = getPosicionCero()
        return 
        
    def __str__(self):
        # Imprime el estado del puzzle
        for l in range(self.lineas):
            print self.estado[l]
            
        print 'Hash Value', self.hashValue
        
        return ''
    
    def __getitem__(self):
        return self.estado
    
    def getLineas(self):
        #Devolvemos el numero de filas del puzzle
        return self.lineas
    
    def getColumnas(self):
        # Devolvemos el numero de columas del puzzle
        return self.columnas
    
    def getPosicionCero(self):
        for i in range(self.lineas):
            for j in range(self.columnas):
                if self.estado[i][j]==0:
                    return i,j
    
    def mhashValue(self):
        vhashValue = ''
        for i in range(self.lineas):
            for j in range(self.columnas):
                vhashValue=vhashValue + str(self.estado[i][j])
                
        return vhashValue
            
    def mGetTile(self,vHashValue):
        m = 0
        
        dimension       = sqrt(len(vHashValue))
        self.lineas     = dimension
        self.columnas   = dimension
        self.hashValue  = vHashValue
        
        for i in range(dimension):
            for j in range(dimension):
                self.estado[i][j] = vHashValue[m]
                m += 1 
        
    def moveTile(self,Direccion):
        """ Mueve el cero en la direccion indicada
            Derecha: 0, Abajo: 1, Izquierda: 2, Arriba: 3"""
        nuevoEstado = self.estado
        x,y = self.getPosicionCero()
        
        if Direccion==0:
            if y < self.columnas-1:
                nuevoEstado[x][y]   = nuevoEstado[x][y+1]
                nuevoEstado[x][y+1] = 0
            else:
                return False
                
        elif Direccion==1:
            if x < self.lineas-1:
                nuevoEstado[x][y]   = nuevoEstado[x+1][y]
                nuevoEstado[x+1][y] = 0
            else:
                return False
            
        elif Direccion==2:
            if y > 0:
                nuevoEstado[x][y]   = nuevoEstado[x][y-1]
                nuevoEstado[x][y-1] = 0
            else:
                return False
                
        elif Direccion==3:
            if x > 0:
                nuevoEstado[x][y]   = nuevoEstado[x-1][y]
                nuevoEstado[x-1][y] = 0
            else:
                return False
            
        self.estado = nuevoEstado
        self.hashValue = self.mhashValue()
        
        return True
    
    def comparaLista(self,lista,vNivel):
        if size(self.estado)==size(lista):
            coste = 0
            distancia = 0
            for i in range(self.lineas):
                for j in range(self.columnas):
                    if self.estado[i][j]<>lista[i][j]:
#                        coste = coste+1
                        coste = coste + abs(self.estado[i][j]-lista[i][j])
                        distancia = distancia + self.estado[i][j]-lista[i][j]
        
        return coste + vNivel #, distancia
               
class cListaSucesores:
    def __init__(self):
        self.lista = {}
        
    def __str__(self):
        #print self.lista
        return str(self.lista)
    
    def __len__(self):
        return len(self.lista)
     
    def addEstado(self,clave,valor):
        self.lista[clave]=valor
        
    def obtenerPadre(self,vEstado):
        try:
            return self.lista[vEstado]
        except:
            return None
        
    def comprobarLista(self,vhashValue):
        # Comprueba si el sucesor ya esta en la lista, si ya es un estado conocido
        return self.lista.has_key(vhashValue)
    
    def imprimirPath(self, vEstado):
        # Dado un estado vEstado imprimir todos sus ancestros hasta el estado inicial
        #while self.lista[vEstado]<>'':
        print vEstado
        if self.lista[vEstado] <> None:
            self.imprimirPath(self.lista[vEstado])
        
    
class cListaCosteTarea:
    def __init__(self):
        self.listaCostes = {}
        
    def __len__(self):
        return len(self.listaCostes)
    
    def __str__(self):
        return str(self.listaCostes)
        
    def addEstado(self, vEstado, vCoste):
        self.listaCostes[vEstado]=vCoste
        
    def delEstado(self,vEstado):
        del self.listaCostes[vEstado]
        
    def getNivelEstado(self,vEstado):
        return self.listaCostes[vEstado]
        
    def proxEstado(self):
        # Reocorre la lista hasta encontrar el estado con el coste minimo
        # para comprobar si es la solucion o calcular sus sucesores
        Coste       = None
        EstadoMin   = ''
        
        for i in self.listaCostes.keys():
            nuevoCoste = self.listaCostes[i]
            if Coste == None or Coste > nuevoCoste:
                Coste       = nuevoCoste
                EstadoMin   = i
        
        return EstadoMin
    
    def noEstadosNivel(self,vNivel):
        # Reocorre la lista hasta encontrar el estado con el coste minimo
        # para comprobar si es la solucion o calcular sus sucesores

        noEstados = 0
        
        for i in self.listaCostes.keys():
            if self.listaCostes[i] == vNivel:
                noEstados  += 1
        
        return noEstados
        