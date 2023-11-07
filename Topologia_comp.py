from itertools import combinations
import networkx as nx
import math

from scipy.spatial import Delaunay,Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
import numpy as np

#PRACTICA 1

#Imprime una lista
def imprimeLista(lista):
    for elem in lista:
        print(elem)

#Ordena una lista de listas
def ordena(lista):
  for elem in lista:
    elem.sort()
  lista.sort()

class Complejo_Simplicial():
    #Lista de tuplas (list, num) donde list representa los simplice maximales (o no maximales pero con un peso distinto al maximal) y num el peso de ese simplice
    complejo_maximal_peso=None

    #CONSTRUCTOR

    # Constructor Definir en Python una clase para almacenar complejos simpliciales.
    # lista_de_simplices es la lista con los simplices, vale con poner los simplices maximales
    def __init__(self, lista_de_simplices, peso=0):
        # Verificamos si lista_de_simplices es una lista y si todos sus elementos son listas. Una lista de listas.
        if isinstance(lista_de_simplices, list) and all(isinstance(simplex, list) for simplex in lista_de_simplices):
            #ordenamos lista de listas
            ordena(lista_de_simplices)
            complejo_maximal = lista_de_simplices
            pesos = [peso]*len(complejo_maximal)
            self.complejo_maximal_peso=list(zip(complejo_maximal,pesos))
            self.complejo_maximal_peso.sort(key=lambda x: (x[1], len(x[0])))
        else:
            raise ValueError("Deben ser todo lista de listas")



    #AUXILIARES

    #Ordena complejo_maximal_peso segun peso (y longitud en caso de empate)
    def ordenaPorPesos(self):
        self.complejo_maximal_peso.sort(key=lambda x: (x[1], len(x[0])))
        return self.complejo_maximal_peso

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

    #Calcula el paseo de un simplice
    def calculaPesoSimplice(self, simplice):
      simplice.sort()
      res = None
      for (simplice_maximal, p) in self.complejo_maximal_peso:
        if res is None and all(elem in simplice_maximal for elem in simplice):
          res = p
        if all(elem in simplice_maximal for elem in simplice):
          res = p if p < res else res
      return res
    
    #Imprime complejo_maximal_peso
    def imprimeSimplicesConPesos(self):
        imprimeLista(self.complejo_maximal_peso)

    #Calcula los pesos de todos los simplices y lo devuelve en una lista de tuplas
    def calculaListaCompletaPesos(self):
        res = []
        simplices = self.caras()
        for simplice in simplices:
            res = res + [(simplice, self.calculaPesoSimplice(simplice))]
        res.sort(key=lambda x: (x[1], len(x[0])))
        return res
    
    #Imprime el resultado de calculaListaCompletaPesos oeri sin los pesos
    def ordenFiltracion(self):
        res = []
        for (simplice, p) in self.calculaListaCompletaPesos():
            res.append(simplice)
        return res

    #Imprime los pesos 
    def PesosOrdenados(self):
        self.ordenaPorPesos()
        res = [0]
        aux = 0
        for (s, p) in self.complejo_maximal_peso:
            if p > aux:
                aux = p
                res.append(p)
        return res


    #PEDIDAS POR LAS PRACTICAS

    # Devuelve la dimension del complejo simplicial, la diemnsión la del simplice mas grande y es el número de vertices menos uno.
    def dimension(self):
        return max(len(simplex) for (simplex,p) in self.complejo_maximal_peso)-1

    # Definir una función que devuelva el conjunto de todas las caras del complejo simplicial.
    def caras(self):
        res=[]
        for (lista,p) in self.complejo_maximal_peso:
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

    # Devuelve todas las n-caras del complejo simplicia
    def carasN(self, n):
        res=[]
        for (lista,p) in self.complejo_maximal_peso:
            combinacion = [list(comb) for comb in combinations(lista, n+1)]
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

    # Devuelve la estrella cerrada del simplice 'simplice'
    def st_cerrada(self, simplice):
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

    # Devuelve el link del simplice 'simplice'
    def link(self, simplice):
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

    #Devuelve el n-equeleto
    def esqueleto(self, n):
        res = []
        for i in range(n+1):
            res.extend(self.carasN(i))
        ordena(res)
        return res

    #Devuelve el numero de componenetes conexas
    def comp_conex(self):
        G = nx.Graph()
        for arista in self.carasN(1):
            G.add_edge(*arista)
        ais = self.aislados()
        for vertice in ais:
            G.add_edge(vertice[0], vertice[0])
        return nx.number_connected_components(G)

    #Añade un nuevo simplice al complejo
    def anadirSimplice(self, simplice, peso):
        if isinstance(simplice, list) and all( (isinstance(simplex, int) or isinstance(simplex, float)) for simplex in simplice):
            simplice.sort()
            self.complejo_maximal_peso.append((simplice,peso))
        else:
            raise ValueError("El simplice dado debe ser una lista de enteros")

    #Añade varios simplices al complejo
    def anadirSimplices(self, lista_de_simplices, peso):
      # Verificamos si lista_de_simplices es una lista y si todos sus elementos son listas. Una lista de listas.
        if isinstance(lista_de_simplices, list) and all(isinstance(simplex, list) for simplex in lista_de_simplices):
            for simplice in lista_de_simplices:
                self.anadirSimplice(simplice, peso)
            #Ordeno la lista del complejo maximal
            self.ordenaPorPesos()
        else:
            raise ValueError("Deben ser todo lista de listas")

    #Devuelve el complejo simplicial con todos los simplices con un peso menor o igual al dado
    def filtracion(self, peso):
        res = Complejo_Simplicial([])
        for (lista, p) in self.complejo_maximal_peso:
            if p>peso:
                break
            res.anadirSimplice(lista, p)
        return res
    
    

#PRACTICA 2

#Calculo el radio de 
def circunrandio(A, B, C):
    #R = (AB* AC* BC)/4*Area
    #Area = det([A0, A1, 1], [B0, B1, 1], [C0, C1, 1])*0.5
    matriz = np.array([[A[0], A[1], 1], [B[0], B[1], 1], [C[0], C[1], 1]])
    area = abs( np.linalg.det(matriz)) * 0.5
    if area == 0:
        raise ValueError("Los siguientes puntos estan alineados: ", A, B, C)

    r = ((math.dist(A,B) * math.dist(A,C) * math.dist(B,C))/area)*0.25

    return r


class AlphaComplex(Complejo_Simplicial):
    #Constructor, se le deben pasar las coordenadas de los puntos en la forma lista de tuplas (o lista) con dos elementos correspondientes a los valores x e y del punto
    def __init__(self, coord):
        self.points = coord
        Del = Delaunay(coord)
        simplices_maximales = [list(elem) for elem in Del.simplices]
        simplices_maximales = [[int(x) for x in sublist] for sublist in simplices_maximales]

        super().__init__([])
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

    #Representa graficamente el alfa complejo para r=peso
    def representaSubnivel(self, peso):
        K = self.filtracion(peso)
        
        fig = voronoi_plot_2d(Voronoi(self.points),show_vertices=False,line_width=2, line_colors='blue' )
        c=np.ones(len(self.points))
        cmap = matplotlib.colors.ListedColormap("limegreen")

        triangulos = K.carasN(2)
        if triangulos:
            plt.tripcolor(self.points[:,0],self.points[:,1],triangulos, c, edgecolor="k", lw=2,cmap=cmap)

        aristas = K.carasN(1)
        for arista in aristas:
            x_arista = [self.points[i, 0] for i in arista]
            y_arista = [self.points[i, 1] for i in arista]
            plt.plot(x_arista, y_arista, color="black", linewidth=2)

        plt.plot(self.points[:,0], self.points[:,1], 'ko')
        plt.show()

    #Representa todo el todos los subcomplejos del alfa-complejo
    def representaAlfaComplejo(self):
        for valor in self.PesosOrdenados():
            self.representaSubnivel(valor)














