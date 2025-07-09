import numpy as np
from scipy.optimize import minimize_scalar

# -----------------------------------------------
# Función objetivo del problema de Fermat-Weber
# -----------------------------------------------

def funcion_objetivo(x, puntos, pesos):

    return np.sum(pesos * np.linalg.norm(puntos - x, axis=1))


# -------------------------------
# Implementacion del Algoritmo de Weiszfeld
# -------------------------------

# -------------------------------
# Operador de Weiszfeld original (T(x))
# -------------------------------

def operador_weiszfeld(x, puntos, pesos):
    
    numerador = np.zeros_like(x)
    denominador = 0.0
    for i in range(len(puntos)):
        distancia = np.linalg.norm(x - puntos[i])
        if distancia == 0:
            continue  # evitar división por cero
        numerador += (pesos[i] * puntos[i]) / distancia
        denominador += pesos[i] / distancia
    return numerador / denominador

# --------------------------------------
# Calcular R(p_j)
# --------------------------------------

def calcular_R(punto_j, puntos, pesos, indice_j):
    
    return sum(pesos[i] * (punto_j - puntos[i]) / np.linalg.norm(punto_j - puntos[i]) 
               for i in range(len(puntos)) if i != indice_j)

# --------------------------------------------------
# Operador S(p_j)
# --------------------------------------------------

def operador_S(punto_j, pesos, puntos, indice_j):
    
    Rj = calcular_R(punto_j, puntos, pesos, indice_j)
    norma_Rj = np.linalg.norm(Rj)
    direccion = -Rj / norma_Rj
    denominador_tj = sum(pesos[i] / np.linalg.norm(puntos[i] - punto_j) 
                         for i in range(len(puntos)) if i != indice_j)
    paso = (norma_Rj - pesos[indice_j]) / denominador_tj
    return punto_j + direccion * paso

# -------------------------------------------------------
# Modificación 2
# -------------------------------------------------------

def punto_inicial(puntos, pesos):

    costos = [np.sum(pesos * np.linalg.norm(puntos[s] - puntos, axis=1)) for s in range(len(puntos))]
    j = np.argmin(costos)
    Rj = calcular_R(puntos[j], puntos, pesos, j)
    if np.linalg.norm(Rj) <= pesos[j]:
        return puntos[j]
    return operador_S(puntos[j], pesos, puntos, j)



# ------------------------------------------------------------------
# Algoritmo completo de Weiszfeld con criterio de parada modificado
# ------------------------------------------------------------------

def weiszfeld(puntos, pesos, tolerancia=1e-6, max_iter=1000):
    
    x = punto_inicial(puntos, pesos)  # Modificación 2

    for iteracion in range(1, max_iter + 1):
        x_nuevo = operador_weiszfeld(x, puntos, pesos)

        # Criterio de parada basado en el gradiente (como en los otros métodos)
        grad = calcular_gradiente(x_nuevo, puntos, pesos)
        if np.linalg.norm(grad) < tolerancia:
            return x_nuevo, iteracion

        x = x_nuevo

    return x, max_iter


# ----------------------------------------------------
# Implementacion del Algoritmo de descenso coordenado
# ----------------------------------------------------

# -----------------------------------------------
# Aproximación del gradiente parcial
# -----------------------------------------------

def gradiente_parcial(x, puntos, pesos, h=1e-8):
    
    gradiente = np.zeros_like(x)
    for i in range(len(x)):
        x_adelante = x.copy()
        x_atras = x.copy()
        x_adelante[i] += h
        x_atras[i] -= h
        gradiente[i] = (funcion_objetivo(x_adelante, puntos, pesos) -
                        funcion_objetivo(x_atras, puntos, pesos)) / (2 * h)
    return gradiente

# -----------------------------------------------
# Descenso coordenado con Gauss-Southwell
# -----------------------------------------------

def descenso_coordenado(puntos, pesos, tolerancia=1e-6, max_iter=1000):
    
    n_dim = puntos.shape[1]
    x = np.mean(puntos, axis=0)  # punto inicial: promedio de los puntos
    iteracion = 0

    while iteracion < max_iter:
        x_anterior = x.copy()
        
         # Calcular el gradiente parcial
        grad = gradiente_parcial(x, puntos, pesos)
        
         # Elegir coordenada de mayor derivada en valor absoluto
        i = np.argmax(np.abs(grad))

         # Optimización univariable sobre la coordenada i
        def funcion_univariable(lamda):
            x_temp = x.copy()
            x_temp[i] += lamda
            return funcion_objetivo(x_temp, puntos, pesos)

        resultado = minimize_scalar(funcion_univariable, method='brent')
        x[i] += resultado.x

        # Verificar convergencia
        if np.linalg.norm(x - x_anterior) < tolerancia:
            break

        iteracion += 1

    return x, iteracion

# ----------------------------------------------------
# Implementacion del Método del Gradiente
# ----------------------------------------------------

# -----------------------------------------------
# Calcular el gradiente de la función objetivo
# -----------------------------------------------

def calcular_gradiente(x, puntos, pesos, epsilon=1e-8):
    
    diferencias = x - puntos
    distancias = np.linalg.norm(diferencias, axis=1).reshape(-1, 1) + epsilon
    
    return np.sum(pesos.reshape(-1, 1) * diferencias / distancias, axis=0)

# -------------------------------------------------
# Método del gradiente usando el criterio de Armijo
# -------------------------------------------------

def metodo_gradiente(puntos, pesos, x_inicial=None, tolerancia=1e-6, c1=1e-4, max_iter=1000):
    
    n = puntos.shape[1]
    
    if x_inicial is None:
        x = np.mean(puntos, axis=0)
        
    else:
        x = np.array(x_inicial, dtype=float)

    
    for iteracion in range(1, max_iter+1):
        gradiente = calcular_gradiente(x, puntos, pesos)
        norma_grad = np.linalg.norm(gradiente)
        
        if norma_grad < tolerancia:
            return x, iteracion
        
        direccion_descenso = -gradiente
        alpha = 1.0  # paso inicial
        valor_actual = funcion_objetivo(x, puntos, pesos)
        
        # Criterio de Armijo
        while funcion_objetivo(x + alpha * direccion_descenso, puntos, pesos) > (valor_actual + c1 * alpha * (gradiente @ direccion_descenso)):
            alpha *= 0.5

        x = x + alpha * direccion_descenso
       

    return x, max_iter