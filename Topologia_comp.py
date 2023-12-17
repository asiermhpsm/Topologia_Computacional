from itertools import combinations
import networkx as nx
import math
import numpy as np

from scipy.spatial import Delaunay,Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation


#FUNCIONES AUXILIARES

#Imprime una lista
def imprimeLista(lista):
    for elem in lista:
        print(elem)

#Convierte una estructura que representa un complejo a la estructura necesaria para la clase Complejo_Simplicial
def normalizaComplejo(sc):
    res = list(sc)
    for i, elem in enumerate(res):
        res[i] = list(elem)
    return res

#Ordena una lista de listas
def ordena(lista):
  for elem in lista:
    elem.sort()
  lista.sort()

#Calculo el radio de circunferencia que pasa por tres puntos
def circunrandio(A, B, C):
    #R = (AB* AC* BC)/4*Area
    #Area = det([A0, A1, 1], [B0, B1, 1], [C0, C1, 1])*0.5
    matriz = np.array([[A[0], A[1], 1], [B[0], B[1], 1], [C[0], C[1], 1]])
    area = abs( np.linalg.det(matriz)) * 0.5
    if area == 0:
        raise ValueError("Los siguientes puntos estan alineados: ", A, B, C)
    return ((math.dist(A,B) * math.dist(A,C) * math.dist(B,C))/area)*0.25


#CLASE PADRE
class Complejo_Simplicial():
    #Lista de tuplas (list, num) donde list representa los simplice maximales 
    #(o no maximales pero con un peso distinto al maximal) y num el peso de ese simplice
    #es decir [([1], 0), ([1,2], 2), ([3,4,5], 3), ...]
    #Además complejo_maximal_peso siempre debe estar ordenado por peso 
    #y en caso de empate por longitud de simplice
    complejo_maximal_peso=None

    #----------------------------------------------------------------
    #CONSTRUCTOR
    #----------------------------------------------------------------

    # lista_de_simplices es la lista con los simplices, vale con poner los simplices maximales
    def __init__(self, lista_de_simplices, peso=0):
        try:
            complejo_maximal = normalizaComplejo(lista_de_simplices)
        except Exception as e:
            raise ValueError("La representación de complejo simplicial debe poder convertirse a lista de listas, pero ha saltado la excepcion ", e)
        
        #ordenamos lista de listas
        ordena(complejo_maximal)
        pesos = [peso]*len(complejo_maximal)
        self.complejo_maximal_peso=list(zip(complejo_maximal,pesos))
        self.complejo_maximal_peso.sort(key=lambda x: (x[1], len(x[0])))

    #----------------------------------------------------------------
    #PRACTICA 1
    #----------------------------------------------------------------

    #--------------------------------
    #Metodos auxiliares
    #--------------------------------

    # Devuelve la estrella cerrada del simplice 'simplice'
    def st_cerrada(self, simplice):
        simplice = list(simplice)
        simplice.sort()
        st = self.st(simplice)
        res = list(st)
        for lista in st:
            #Obtengo todas las posibles combinaciones de todas las longitudes posibles
            for i in range(len(lista)):
                combinacion = [list(comb) for comb in combinations(lista, i+1)]
                for elem in combinacion:
                    #si el elemento (ordenado) no esta ya añadido, lo añado
                    elem.sort()
                    if elem not in res:
                        res.append(elem)
        ordena(res)
        return res

    #Devuelve True si un simplice es cara de otro
    def esCara(self, simpliceMayor, simpliceMenor):
        return all(elem in simpliceMayor for elem in simpliceMenor)

    #Devuelve los vértices aislados
    def aislados(self):
        res = []
        for (elem,p) in self.complejo_maximal_peso:
            if len(elem) == 1:
                cont = 0
                for (simp,p) in self.complejo_maximal_peso:
                    if elem[0] in simp:
                        cont +=1
                        if cont > 1:
                            break
                if cont < 2:
                    res.append(elem)
        ordena(res)
        return res

    #Devuelve el n-equeleto
    def esqueleto(self, n):
        res = []
        for i in range(n+1):
            res.extend(self.carasN(i))
        ordena(res)
        return res

    #Calcula el paseo de un simplice, devuelve None si el simplice no existe
    def calculaPesoSimplice(self, simplice):
      simplice.sort()
      res = None
      for (simplice_maximal, p) in self.complejo_maximal_peso:
        if res is None and all(elem in simplice_maximal for elem in simplice):
          res = p
        if all(elem in simplice_maximal for elem in simplice):
          res = p if p < res else res
      return res

    #Calcula el paseo de un simplice, que será el minimo de los maximales en los que esté
    def calculaPesoSimplice(self, simplice):
      simplice.sort()
      res = None
      for (simplice_maximal, p) in self.complejo_maximal_peso:
        if res is None and self.esCara(simplice_maximal, simplice):
          res = p
        elif self.esCara(simplice_maximal, simplice):
          res = p if p < res else res
      return res

    #Añade un nuevo simplice al complejo
    def anadirSimplice(self, simplice, peso):
        simplice = list(simplice)
        #if isinstance(simplice, list) and all( (isinstance(simplex, int) or isinstance(simplex, float)) for simplex in simplice):
        if isinstance(simplice, list) and all( isinstance(simplex, int) for simplex in simplice):
            simplice.sort()
            peso_actual = self.calculaPesoSimplice(simplice)
            #Solo añado el simplice si el simplice no existe
            #o el nuevo peso es menor que el que tenía antes
            if peso_actual is None or peso < peso_actual:
                self.complejo_maximal_peso.append((simplice,peso))   
        else:
            raise ValueError("El simplice dado debe ser una lista de enteros")

    #Devuelve el orden en el que aparecen los simplices en una filtracion
    def ordenFiltracion(self):
        res = []
        for (simplice, _) in self.calculaListaCompletaPesos():
            res.append(simplice)
        return res

    #--------------------------------
    #Métodos pedidos
    #--------------------------------

    # Devuelve la dimension del complejo simplicial, la diemnsión la del simplice mas grande y es el número de vertices menos uno.
    def dimension(self):
        return max(len(simplex) for (simplex,_) in self.complejo_maximal_peso)-1

    # Devuelve el conjunto de todas las caras del complejo simplicial.
    def caras(self):
        res=[]
        for (simplice,_) in self.complejo_maximal_peso:
            #Obtengo todas las posibles combinaciones de todas las longitudes posibles
            for i in range(len(simplice)):
                combinacion = [list(comb) for comb in combinations(simplice, i+1)]
                for elem in combinacion:
                    #si el elemento (ordenado) no está ya añadido, lo añado
                    elem.sort()
                    if elem not in res:
                        res.append(elem)
        ordena(res)
        return res

    # Devuelve todas las n-caras del complejo simplicia
    def carasN(self, n):
        res=[]
        for (simplice,_) in self.complejo_maximal_peso:
            combinacion = [list(comb) for comb in combinations(simplice, n+1)]
            for elem in combinacion:
                #si el elemento (ordenado) no esta ya añadido, lo añado
                elem.sort()
                if elem not in res:
                    res.append(elem)
        ordena(res)
        return res

    # Devuelve la estrella del simplice 'simplice'
    def st(self, simplice):
        complejo_simplicial=self.caras()
        simplice = list(simplice)
        simplice.sort()
        if simplice not in complejo_simplicial:
            raise ValueError("El simplice dado no se encuentra en el complejo simplicial")
        else:
            res = []
            for simplice_aux in complejo_simplicial:
                if self.esCara(simplice_aux, simplice):
                    res.append(simplice_aux)
            ordena(res)
            return res

    # Devuelve el link del simplice 'simplice'
    def link(self, simplice):
        simplice = list(simplice)
        simplice.sort()
        st_c = self.st_cerrada(simplice)
        res = []
        for elem_c in st_c:
            if all(elem not in elem_c for elem in simplice):
                res.append(elem_c)
        ordena(res)
        return res

    #Devuelve la característica de Euler
    def euler(self):
        res = 0
        for i in range(self.dimension()+1):
            res += len(self.carasN(i))*(-1)**i
        return res

    #Devuelve el numero de componenetes conexas
    def comp_conex(self):
        G = nx.Graph()
        #Creo un grafo con el esqueleto
        for arista in self.carasN(1):
            G.add_edge(*arista)
        #Los puntos aislados los añado creando aristas con ellos como principio y como fin
        ais = self.aislados()
        for vertice in ais:
            G.add_edge(vertice[0], vertice[0])
        return nx.number_connected_components(G)

    #Añade varios simplices al complejo
    def anadirSimplices(self, lista_de_simplices, peso):
        lista_de_simplices = normalizaComplejo(lista_de_simplices)
      # Verificamos si lista_de_simplices es una lista y si todos sus elementos son listas. Una lista de listas.
        if isinstance(lista_de_simplices, list) and all(isinstance(simplex, list) for simplex in lista_de_simplices):
            for simplice in lista_de_simplices:
                self.anadirSimplice(simplice, peso)
            #Ordeno la lista del complejo maximal
            self.complejo_maximal_peso.sort(key=lambda x: (x[1], len(x[0])))
        else:
            raise ValueError("Deben ser todo lista de listas")

    #Devuelve el complejo simplicial con todos los simplices con un peso menor o igual al dado
    def filtracion(self, peso):
        res = Complejo_Simplicial([])
        for (lista, p) in self.complejo_maximal_peso:
            if p<=peso:
                res.anadirSimplice(lista, p)  
        return res


    #----------------------------------------------------------------------------------------------------------------------
    #PRACTICA 2 (parte 2)
    #----------------------------------------------------------------------------------------------------------------------

    #--------------------------------
    #Métodos auxiliares
    #--------------------------------

    #Generalización de representación de complejo según un peso,
    #las clases hijas podrán añadir algo mas si es necesario
    #axes simplemente para representarlo en una cierta ventana de matplotlib así se desea
    def representaSubnivel(self, peso, ax=None):
        try:
            self.points
        except Exception as e:
            raise ValueError("No se puede representar un complejo simplicial directamente desde la clase padre")
        
        if ax is None:
            fig, ax = plt.subplots()

        K = self.filtracion(peso)
        
        c=np.ones(len(self.points))
        cmap = matplotlib.colors.ListedColormap("limegreen")

        triangulos = K.carasN(2)
        if triangulos:
            ax.tripcolor(self.points[:,0],self.points[:,1],triangulos, c, edgecolor="k", lw=2,cmap=cmap)

        aristas = K.carasN(1)
        for arista in aristas:
            x_arista = [self.points[i, 0] for i in arista]
            y_arista = [self.points[i, 1] for i in arista]
            ax.plot(x_arista, y_arista, color="black", linewidth=2)

        ax.plot(self.points[:,0], self.points[:,1], 'ko')

    #Representa los puntos
    def representaPuntos(self, ax=None):
        try:
            self.points
        except Exception as e:
            raise ValueError("No se puede representar un complejo simplicial directamente desde la clase padre")
        
        mostrar = False
        if ax is None:
            mostrar = True
            fig, ax = plt.subplots()
        
        ax.scatter(self.points[:,0], self.points[:,1], color='black', marker='o')

        if mostrar:
            plt.show()

    #--------------------------------
    #Métodos pedidos
    #--------------------------------

    #Representa todo el todos los subcomplejos del alfa-complejo
    def representaComplejo(self):
        for valor in self.PesosOrdenados():
            self.representaSubnivel(valor)
            plt.show()


    #----------------------------------------------------------------------------------------------------------------------
    #PRACTICA 3
    #----------------------------------------------------------------------------------------------------------------------

    #--------------------------------
    #Métodos auxiliares
    #--------------------------------

    #Dada una matriz en la que matriz[i][i] != 1, intercambia una fila o una columna para conseguirlo el 1
    def consigueUno(self, matriz, i):
        for j in range(i, len(matriz)):
            for k in range(i, len(matriz[0])):
                if matriz[j][k] == 1:
                    matriz[i], matriz[j] = matriz[j], matriz[i]
                    matriz = self.cambiaColum(matriz, i, k)
                    return matriz, False
        return matriz, True

    #Dada una matriz con un 1 en la posicion matriz[i][i], consigue todo 0's en la columna i a partir de la fila i+1
    def despejaColumna(self, matriz, i):
        for cont in range(i+1, len(matriz)):
            if matriz[cont][i] == 1:
                self.sumoFila(matriz, cont, i)
        return matriz
    
    #Dada una matriz con un 1 en la posicion matriz[i][i], consigue todo 0's en la fila i a partir de la columna i+1
    def despejaFila(self, matriz, i):
        traspuesta = self.traspuesta(matriz)
        self.despejaColumna(traspuesta, i)
        return self.traspuesta(traspuesta)
    
    #Intecambia columna i por fila j
    def cambiaColum(self, matriz, i, j):
        traspuesta = self.traspuesta(matriz)
        traspuesta[i], traspuesta[j] = traspuesta[j], traspuesta[i]
        return self.traspuesta(traspuesta)

    #Devuelve la matriz traspuesta
    def traspuesta(self, matriz):
        return [[fila[i] for fila in matriz] for i in range(len(matriz[0]))]

    #Dada una matriz a la fila i le sumo la fila j modulo 2
    def sumoFila(self, matriz, i, j):
        for k in range(len(matriz[0])):
            matriz[i][k] = (matriz[i][k] +  matriz[j][k])%2
    
    #Dada una matriz normal de Smith, devuelve la cantidad de 1's que hay
    def longDiagonal(self, matriz):
        cont = 0
        for i in range(min(len(matriz), len(matriz[0]))):
            if matriz[i][i] == 1:
                cont += 1
        return cont
    

    #--------------------------------
    #Métodos pedidos
    #--------------------------------

    #Devuelve la matriz brode-p
    def matrizBorde(self, p):
        if not isinstance(p, int):
            raise ValueError("El p dado debe ser un entero")
        elif p < 0 or p > self.dimension():
            raise ValueError("El p dado debe estar entre 0 y la dimension del complejo simplicial")
        elif p == 0:
            return [[0] * len(self.carasN(0)) for _ in range(1)]
        else:
            leyenda_filas = self.carasN(p-1)
            leyenda_columnas = self.carasN(p)
            matriz = [[0] * len(leyenda_columnas) for _ in range(len(leyenda_filas))]
            
            for i in range(len(leyenda_filas)):
                for j in range(len(leyenda_columnas)):
                    if self.esCara(leyenda_columnas[j], leyenda_filas[i]):
                        matriz[i][j] = 1
            return matriz

    #Dada una matriz (lista de listas) devuelve la forma normal de Smith
    def formaNormalSmith(self, p):
        matriz = self.matrizBorde(p)
        for i in range(min(len(matriz), len(matriz[0]))):
            if matriz[i][i] != 1:
                matriz, fin = self.consigueUno(matriz, i)
                if fin:
                    break
            matriz = self.despejaColumna(matriz, i)
            matriz = self.despejaFila(matriz, i)

        return matriz

    #Devuelve el numero de Betti
    def numeroBetti(self, p):
        normSimth_p = self.formaNormalSmith(p)
        Z_p = len(normSimth_p[0]) - self.longDiagonal(normSimth_p)
        if p == self.dimension():
            B_p = 0
        else:
            B_p = self.longDiagonal(self.formaNormalSmith(p+1))        
        return Z_p - B_p

    #Devuelve el numero de Betti mediante el algoritmo incremental
    def numBettiIncremental(self):
        N_iMenos1 = self.filtracion(0)
        simplice_nuevo = None
        beta0 = 0
        beta1 = 0
        for valor in self.PesosOrdenados():
            if valor == 0 :
                beta0 = len(self.carasN(0))
            else:
                N_i = self.filtracion(valor)
                for elem,p in self.complejo_maximal_peso:
                    if p == valor:
                        simplice_nuevo = elem

                if len(simplice_nuevo)==1:
                    beta0 = beta0 + 1
                elif len(simplice_nuevo)==2:
                    if N_i.comp_conex == N_iMenos1.comp_conex:
                        beta1 = beta1 + 1
                    else:
                        beta0 = beta0 - 1
                else:
                    beta1 = beta1 - 1
                N_iMenos1 = N_i
            #print("Complejo: ", N_iMenos1.complejo_maximal_peso)
            #print("Componentes conexas: ", N_iMenos1.comp_conex())
            #print(beta0, beta1, "\n")
        return beta0, beta1


    #----------------------------------------------------------------------------------------------------------------------
    #PRACTICA 4
    #----------------------------------------------------------------------------------------------------------------------

    #--------------------------------
    #Métodos auxiliares
    #--------------------------------

    #Devuelve el ultimo putno de persistenca de dgm1 que no nace y muere en el mismo momento
    def limite_representacion(self, dgm1):
        for (nacimiento, muerte) in dgm1[::-1]:
            if muerte-nacimiento != 0:
                return int(muerte) + 1
        return 0

    #--------------------------------
    #Métodos pedidos
    #--------------------------------

    #Devuelve la lista de puntos del diagrama de persistencia
    def ptosPersistencia(self):
        #Calculo el alfa-complejo y todos simplices con sus respectivos pesos
        carasPesos = self.calculaListaCompletaPesos()
        caras = []
        for (simplice, _) in carasPesos:
            caras.append(simplice)

        #Calculo la matriz del algoritmo
        N = len(caras)
        matriz = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if len(caras[i]) == len(caras[j])-1 and self.esCara(caras[j], caras[i]):
                    matriz[i][j] = 1

        dgm0 = []
        dgm1 = []
        lows = []
        j = 0
        while(j<N):
            #Calculo el 1 mas abajo (-1 si es una columna de 0s)
            k = -1
            for i in range(N):
                if matriz[i][j] == 1:
                    k = i
            #Si es una columna de 0s sigo a la siguente columna
            if k == -1:
                j = j + 1
                continue
            
            #Hago al algoritmo para encontrar el low
            lowEncontrado = True
            #Si el 1 mas abajo esta a la altura de algun low entonces sumo columnas
            for (fil, col) in lows:
                if fil == k:
                    for l in range(N):
                        matriz[l][j] = (matriz[l][j] + matriz[l][col])%2
                    lowEncontrado = False
                    break

            #Si he encontrado el low lo añado a los lows,  calculo el dgm0 o dgm1 y paso a la siguiente columna
            if lowEncontrado:
                lows.append([k, j])
                if len(carasPesos[k][0])==1:
                    dgm0.append((0, carasPesos[j][1]))
                elif len(carasPesos[k][0])==2:
                    dgm1.append((carasPesos[k][1], carasPesos[j][1]))
                j = j + 1
        
        #Añado manualmente el punto del infinito
        dgm0.append((0, int(max(dgm0 + dgm1, key=lambda x: x[1])[0])+1))

        return dgm0, dgm1

    #Muestra el diagrama de persistencia
    def diagramaPersistencia(self, ax=None):        
        mostrar = False
        if ax is None:
            mostrar = True
            fig, ax = plt.subplots()

        dgm0, dgm1 = self.ptosPersistencia()
        ax.plot([tupla[0] for tupla in dgm0],[tupla[1] for tupla in dgm0],'bo')
        ax.plot([tupla[0] for tupla in dgm1],[tupla[1] for tupla in dgm1],'ro')
        maximo = max(dgm0[-1][1], dgm1[-1][-1])
        x_values = np.linspace(0, maximo, 100)
        ax.plot(x_values, x_values, 'k--')
        ax.plot(x_values, [maximo]*100, 'k--')
        ax.set_xlabel('Nacimiento')
        ax.set_ylabel('Muerte')
        ax.set_xlim(-0.05, self.limite_representacion(dgm1))
        ax.set_ylim(-0.05, self.limite_representacion(dgm1))

        if mostrar:
            plt.show()

    #Muestra el codigo de barras
    def codigoBarras(self, ax=None):
        mostrar = False
        if ax is None :
            mostrar = True
            fig, ax= plt.subplots()

        dgm0, dgm1 = self.ptosPersistencia()
        k=0
        separacion = self.limite_representacion(dgm1)/10
        for i, (inicio, fin) in enumerate(dgm0):
            ax.plot([inicio, fin], [i+separacion, i+separacion], color='blue', linewidth=1)
            k=i
        for i, (inicio, fin) in enumerate(dgm1):
            ax.plot([inicio, fin], [k+1+i+separacion, k+1+i+separacion], color='red', linewidth=1)
            
        ax.set_xlim(0, self.limite_representacion(dgm1))
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_yaxis().set_visible(False)

        if mostrar:
            plt.show()


    #----------------------------------------------------------------------------------------------------------------------
    #METODOS AUXILIARES GENERALES
    #----------------------------------------------------------------------------------------------------------------------

    #Calcula los pesos de todos los simplices y lo devuelve en una lista de tuplas
    def calculaListaCompletaPesos(self):
        res = []
        simplices = self.caras()
        for simplice in simplices:
            res = res + [(simplice, self.calculaPesoSimplice(simplice))]
        res.sort(key=lambda x: (x[1], len(x[0])))
        return res
    
    #Devuelve la lista ordenada de todos los pesos que hay en el complejo simplicial
    def PesosOrdenados(self):
        self.complejo_maximal_peso.sort(key=lambda x: (x[1], len(x[0])))
        res = [0]
        aux = 0
        for (s, p) in self.complejo_maximal_peso:
            if p > aux:
                aux = p
                res.append(p)
        return res
    

    #----------------------------------------------------------------------------------------------------------------------
    # AMPLIACION DE REPRESENTACION
    #----------------------------------------------------------------------------------------------------------------------
    
    #Representa el complejo en una animacion
    def animaComplejo(self, fig=None, ax=None):
        mostrar = False
        if ax is None or fig is None:
            mostrar = True
            fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            peso = self.PesosOrdenados()[frame]
            self.representaSubnivel(peso, ax)
            ax.set_title('Triangulacion')

        num_frames = len(self.PesosOrdenados())
        ani = FuncAnimation(fig, update, frames=num_frames, interval=int(len(self.caras())/10))

        if mostrar:
            plt.show()
        else:
            return ani

    #Presenta resultados
    def analiza(self):
        fig, axes = plt.subplots(2, 2)

        self.representaPuntos(ax=axes[0, 0])
        axes[0, 0].set_title('Puntos')

        ani = self.animaComplejo(fig, axes[0, 1])

        self.diagramaPersistencia(axes[1, 0])
        axes[1, 0].set_title('Diagrama de persistencia')

        self.codigoBarras(axes[1, 1])
        axes[1, 1].set_title('Codigo de barras')

        plt.tight_layout()
        plt.show()


#----------------------------------------------------------------------------------------------------------------------
#PRACTICA 2 (parte 1)
#----------------------------------------------------------------------------------------------------------------------

#CLASE ALFA COMPLEJO, es clase hija de 'Complejo_Simplicial'
class AlphaComplex(Complejo_Simplicial):
    #Coordenadas de los puntos
    points = None

    #Constructor, se le deben pasar las coordenadas de los puntos como se pide hace en los ejemplos de clase
    def __init__(self, coord):
        self.points = coord
        Del = Delaunay(coord)
        #Fuerzo la conversion a lista de listas con enteros (tipos nativos de python, no de numpy)
        simplices_maximales = [list(elem) for elem in Del.simplices]
        simplices_maximales = [[int(x) for x in sublist] for sublist in simplices_maximales]
        ordena(simplices_maximales)

        #Primero se inicializa el complejo vacío, luego se irán añadiendo los simplice a medida que se calculo su peso
        super().__init__([])

        #Variable auxiliar para conseguir los simplice del complejo simplicial
        sc_aux = Complejo_Simplicial(simplices_maximales)

        self.anadirPesosVertices(sc_aux.carasN(0))
        self.anadirPesosTriangulos(coord, sc_aux.carasN(2))
        self.anadirPesosAristas(coord, sc_aux.carasN(1), sc_aux.carasN(0))

    #Añade los pesos de los vertices
    def anadirPesosVertices(self, vertices):
        for vertice in vertices:
            self.anadirSimplice(vertice, 0)

    #Añade los pesos de los triangulos
    def anadirPesosTriangulos(self, coord, triangulos):
        for triangulo in triangulos:
            p1 = coord[triangulo[0]]
            p2 = coord[triangulo[1]]
            p3 = coord[triangulo[2]]
            radio = circunrandio(p1, p2, p3)
            self.anadirSimplice(triangulo, radio)

    #Añade los pesos de las aristas
    def anadirPesosAristas(self, coord, aristas, vertices):
        for arista in aristas:
                ptoDentroCirc = False
                p1 = coord[arista[0]]
                p2 = coord[arista[1]]
                radio = math.dist(p1,p2)*0.5
                x = (p1[0]+ p2[0])/2
                y = (p1[1]+ p2[1])/2
                for vertice in vertices:
                    if vertice[0] != arista[0] and vertice[0] != arista[1]:
                        if math.dist((x, y), coord[vertice[0]]) <= radio:
                            ptoDentroCirc = True
                            break
                if not ptoDentroCirc:
                    self.anadirSimplice(arista, radio)

    #Se entra en detalle de la generalización de la clase padre
    #ax sirve para representarlo en una cierta ventana de matplotlib así se desea
    def representaSubnivel(self, peso, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        voronoi_plot_2d(Voronoi(self.points),ax=ax, show_vertices=False,line_width=2, line_colors='blue' )
        super().representaSubnivel(peso, ax=ax)


#----------------------------------------------------------------------------------------------------------------------
#PRACTICA 2 (parte 3)
#----------------------------------------------------------------------------------------------------------------------
#COMPLEJO VIETORI-RIPS
class VietorisRips(Complejo_Simplicial):
    #Constructor 
    def __init__(self, coord):
        self.points = coord

        super().__init__([])
        sc_aux = Complejo_Simplicial([list(range(len(coord)))])

        #Añado peso de vertice
        for vertice in sc_aux.carasN(0):
            self.anadirSimplice(vertice, 0)

        #Añado peso aristas
        for arista in sc_aux.carasN(1):
            p1 = coord[arista[0]]
            p2 = coord[arista[1]]
            self.anadirSimplice(arista, math.dist(p1, p2)*0.5)

        #Añado peso del resto de simplices
        for i in range(2, sc_aux.dimension()+1):
            for i_simplice in sc_aux.carasN(i):
                combinacion = [list(comb) for comb in combinations(i_simplice, i)]
                pesos = [self.calculaPesoSimplice(cara) for cara in combinacion]
                self.anadirSimplice(i_simplice, max(pesos))


        


    





