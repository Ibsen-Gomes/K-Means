############################################ Código K-means (machine learning) ##################################################

# Bibliotecas
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import sys
import random
from math import sqrt


# Distância Euclidiana:
def euclidian(v1, v2, check_input=True):
    
    '''
    Esse código revebe dois vetores e calcula a distancia euclidiana entre vetores
    
    entrada
    v1 = lista 1D
    v2 = lista 1D
    
    saída
    distancia euclidiana
    '''
    
    # Assert para assegurar que as entradas são 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
    
    #Armazena o quadrado da distância
    dist = 0.0
    for x in range(len(v1)):
        dist += (v1[x] - v2[x])**2
 
    #Tira a raiz quadrada da soma
    eucli = sqrt(dist)
    return eucli


# Distância Manhattan:
def manhattan(v1, v2, check_input=True):
    
    '''
    Esse código revebe dois vetores e calcula a distancia Manhattan entre vetores
    
    entrada
    v1 = lista 1D
    v2 = lista 1D
    
    saída
    distancia manhattan
    '''
    
    # Assert para assegurar que as entradas são 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
    
    #Armazena a distância simples
    dist = 0.0
    for x in range(len(v1)):
        dist += (v1[x] - v2[x])
 
    #atribui dist para a variável de saida manhattan
    manhattan = dist
    return manhattan


# inicialização kmeans++:
def kmeans_plus_plus(data, k):
    
    '''
    Inicialização usando o kmeans++:
    inputs:
    data = numpy array onde será sorteado um ponto inicial e a partir dele selecionar outros centroides
    k = numero de clusters
    
    output:
    coordenadas dos centroides após o uso do kmeans++
    
    Obs: o código não seleciona o próximo centroide através de uma distribuição de probabilidade
    ele apenas seleciona o proximo centroide como sendo o mais distante dos centroides ja selecionados!
    '''
    
    ## inicializa uma lista de centróides e adiciona um ponto de dados selecionado aleatoriamente para a lista:
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
  
    ## iterações para gerar os outros centroides (k - 1):
    for c_id in range(k - 1):
         
        ## inicializa uma lista para armazenar distâncias entre os dados até o centróide mais próximo:
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
             
            ## calcula a distância de um dado até os centróide, selecionado e armazene a distância mínima:
            for j in range(len(centroids)):
                temp_dist = euclidian(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## seleciona um dado com distância máxima como nosso próximo centróide
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        
    return centroids


# K-means:
def K_means(data, k, it, tol, random='range'):
    
    '''
    Esse código aplica o método k-means a um conjunto de dados de entrada
    
    entrada:
    data = dados a serem agrupados;
    k = n° de centroides (quantos grupos se quer dividir os dados)
    tol = tolerância 
    random = forma de inicialização dos centroides
    
    método:
    -> calculo da distância entre cada dado com os centroides
    -> atualização da posição do centroide para a média do grupo
    -> se posição atual - posição anterios dos centroides for menor que tolerância (tol) o código para
    -> se o critério de tolerância não for atendida, o código continua a realizar as iterações (talvez faça todas as iterações!)
    
    saída:
    bestmatches = dados separados em grupos
    centroids = coordenadas dos centroides após o fim das iterações 
    conta = quantas iterações foram realizadas
    '''
     
    centroids = np.zeros((k,2)) # Array vazio que será preencido pelos Centroids
    
    # Condicional -> se "range", os centroides serão encolhidos de forma aleatória dentro de um range:
    if random == 'range':
        for i in range(k):
            x = np.array(np.random.uniform(min(data[:,0]),max(data[:,0])))
            y = np.array(np.random.uniform(min(data[:,1]),max(data[:,1])))
            centroids[i] = x, y
            
    # Condicional -> se "input_data", os centroides serão escolhidos entre os dados de entrada:     
    if random == 'input_data':
        for i in range(k):
            index = np.random.randint(data.shape[0])
            centroids[i] = data[index,:]
            
    # Condicional -> para inicialização k-means++:
    if random == 'kmeans++':
        centroids = kmeans_plus_plus(data, k) # chamando a função kmeans++
        
    #tol = 1
    conta = 0
    lastmatches = None
    #O número de iterações será no máximo 100
    for t in range(it):
        bestmatches = [[] for i in range(k)]
     
        #Verifica qual centroide esta mais perto de cada instancia
        for j in range(len(data)):
            row=data[j]
            bestmatche = 0 #Aqui armazeno o índice da menor distância para comparação
            for i in range(k):
                d = euclidian(centroids[i],row) #Calcula a distancia em relação ao centroide
                if d < euclidian(centroids[bestmatche],row): ###Comparação entre as distâncias entre um dado e todos centroides?
                    bestmatche = i
            bestmatches[bestmatche].append(j)
     
        #Move o centroide para a zona média do cluster
        #no caso teremos 
        for i in range(k):
            avgs=[0.0]*len(data[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(data[rowid])):
                        avgs[m] += data[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                centroids[i]=avgs
                
        # Condicional para convergência: se a posição atual dos centroides - posição anterior
        # for menor que uma tolerância o código para e converge!
        if np.allclose(centroids[i],centroids[i-1], rtol=tol) == True:
            break 
            
        lastmatches=bestmatche
        conta += 1 # conta quantas iterações ocorreram
        
    return bestmatches, centroids, conta 
