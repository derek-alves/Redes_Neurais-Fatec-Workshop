import numpy as np
from random import uniform, randint

def le_arquivo(arq):
    f = open(arq,"r")

    dados_str = []
    for line in f:
        dados_str.append(line.strip('\n').split(','))
    f.close()
    return dados_str


def rna_treino(data, TMAX, nx, ns):
    alfa = 0.4  # constante de treinamento = a cada interação qual vai ser a mudança
    x = np.zeros(nx+1, float)  # nos da camada de entrada
    v = np.zeros(nx+1, float)  # pesos de entrada pra saida
    for i in range(nx+1):  # inicializando os pesos
        v[i] = uniform(-1, 1)  # com valores entre -1 e +1

    for t in range(TMAX):
        line = randint(0, ns-1)
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1
        se = int(data[line][nx])  # saida esperada

        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if(y >= 0):  # saida real
            sr = +1
        else:
            sr = -1
        if(se != sr):  # teste para se as duas saidas são diferentes
            for i in range(nx+1):
              # se entrar nesse if quer dizer que a rede
              # errou comparando o resultado gerado pelo algoritmo com o resultado esperado
              # necessitando de uma alteração no peso
                v[i] += alfa*(se - sr)*x[i]
    return v


def rna_teste(data, TMAX, nx, ns):
    x = np.zeros(nx+1, float)  # nos da camada de entrada
    qa = 0

    for line in range(ns):  # inicializando os pesos
        for i in range(nx):
            x[i] = data[line][i]
        x[nx] = 1
        se = int(data[line][nx])

        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if(y >= 0):
            sr = +1
        else:
            sr = -1
        if(se == sr):
            qa += 1

    return qa*100./ns

# PROGRAMA PRINCIPAL
dados = []
dados = le_arquivo("iris_perceptron.txt")
nx = len(dados[0])-1  # qtd. camada de entrada
ns = len(dados)  # qtd. dados treino
# qtd. de interações a rede neural vai ter
tmax = [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
qt = 30  # todo ciclo vai gerar uma precisão e todas elas vão ser somadas e dividida por essa variável

for i in range(len(tmax)):
    prec = 0
    for j in range(len(tmax)):
        v = rna_treino(dados, tmax[i], nx, ns)
        prec += rna_teste(dados, v, nx, ns)
    print("TMAX = ", tmax[i], "Precisão = ", prec/qt)
