from Implementaciones import crear_instancia, leer_instancia, funcion_objetivo, weiszfeld, descenso_coordenado, metodo_gradiente
import time

crear_instancia(nombre_archivo="instancia_18.txt", n_dim=50, n_puntos = 10000,
                    rango_coords=(-50000, 50000), rango_pesos=(1, 1), distribucion="uniforme", seed=None)

puntos, pesos = leer_instancia("instancia_18.txt")

# -----------------------------------------------
# Evaluar Algoritmo de Weiszfeld
# -----------------------------------------------

tiempo_inicio_weiszfeld = time.time()
solucion_weiszfeld, iteraciones_weiszfeld = weiszfeld(puntos, pesos)
tiempo_fin_weiszfeld = time.time()
tiempo_total_weiszfeld = tiempo_fin_weiszfeld- tiempo_inicio_weiszfeld
valor_en_funcion_objetivo_weiszfeld = funcion_objetivo(solucion_weiszfeld, puntos, pesos)

# -----------------------------------------------
# Evaluar Método Descenso Coordenado
# -----------------------------------------------

tiempo_inicio_des_coord = time.time()
solucion_des_coord, iteraciones_des_coord = descenso_coordenado(puntos, pesos)
tiempo_fin_des_coord = time.time()
tiempo_total_des_coord = tiempo_fin_des_coord - tiempo_inicio_des_coord
valor_en_funcion_objetivo_des_coord = funcion_objetivo(solucion_des_coord, puntos, pesos)


# -----------------------------------------------
# Evaluar Método Gradiente
# -----------------------------------------------

tiempo_inicio_gradiente = time.time()
solucion_gradiente, iteraciones_gradiente = metodo_gradiente(puntos = puntos, pesos = pesos)
tiempo_fin_gradiente= time.time()
tiempo_total_gradiente = tiempo_fin_gradiente - tiempo_inicio_gradiente
valor_en_funcion_objetivo_gradiente = funcion_objetivo(solucion_gradiente, puntos, pesos)


print("Valores Obtenidos Por el Algoritmo de Weiszfeld usando Modificacion 2: \n")

print("Centro óptimo Weiszfeld:", solucion_weiszfeld)
print(f"Iteraciones Weiszfeld (Tiempo de Convergencia): {iteraciones_weiszfeld}")
print("Valor función objetivo Weiszfeld:", valor_en_funcion_objetivo_weiszfeld)
print(f"Tiempo de Ejecucion: {tiempo_total_weiszfeld}")

print("\n")

print("Valores Obtenidos Por el Método de Descenso Coordenado: \n")

print("Centro óptimo Descenso Coordenado:", solucion_des_coord)
print(f"Iteraciones Descenso Coordenado (Tiempo de Convergencia): {iteraciones_des_coord}")
print("Valor función objetivo Descenso Coordenado:", valor_en_funcion_objetivo_des_coord)
print(f"Tiempo de Ejecucion Descenso Coordenado: {tiempo_total_des_coord}")

print("\n")

print("Valores Obtenidos Por el Método del Gradiente: \n")

print("Centro óptimo Gradiente:", solucion_gradiente)
print(f"Iteraciones Gradiente (Tiempo de Convergencia): {iteraciones_gradiente}")
print("Valor función objetivo Gradiente:", valor_en_funcion_objetivo_gradiente)
print(f"Tiempo de Ejecucion Gradiente: {tiempo_total_gradiente}")