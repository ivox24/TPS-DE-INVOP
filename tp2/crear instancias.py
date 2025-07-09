import random
import math
import heapq
from collections import defaultdict
import numpy as np

VALOR_GRANDE = 10**6

# --------------------------
# 1. Generar instancia inicial
# --------------------------

def generar_instancia(nombre_archivo, cant_clientes=20, costo_repartidor=5, dist_max=50,
                      cant_refrigerados=7, cant_exclusivos=11, porcentaje_conectividad=0.1, rango_coord=100, tipo_distribucion='uniforme'):
    ids_clientes = list(range(1, cant_clientes + 1))
    refrigerados = sorted(random.sample(ids_clientes, cant_refrigerados))
    restantes = list(set(ids_clientes) - set(refrigerados))
    exclusivos = sorted(random.sample(restantes, min(cant_exclusivos, len(restantes))))
    
    # Selección de distribución
    if tipo_distribucion == 'uniforme':
        coords = generar_coords_uniforme(cant_clientes, rango_coord)
    elif tipo_distribucion == 'clusters':
        coords = generar_coords_cluster(cant_clientes, rango_coord)
    elif tipo_distribucion == 'anillo':
        coords = generar_coords_anillo(cant_clientes, rango_coord)
    else:
        coords = {i: (random.randint(0, rango_coord), random.randint(0, rango_coord)) for i in [0] + ids_clientes}

    with open(nombre_archivo, 'w') as f:
        f.write(f"{cant_clientes}\n")
        f.write(f"{costo_repartidor}\n")
        f.write(f"{dist_max}\n")
        f.write(f"{cant_refrigerados}\n")
        for r in refrigerados:
            f.write(f"{r}\n")
        f.write(f"{len(exclusivos)}\n")
        for e in exclusivos:
            f.write(f"{e}\n")

        aristas = {}
        nodos = [0] + ids_clientes
        for i in nodos:
            for j in nodos:
                if i < j and random.random() < porcentaje_conectividad:
                    xi, yi = coords[i]
                    xj, yj = coords[j]
                    dist = round(math.hypot(xi - xj, yi - yj))
                    costo = dist
                    aristas[(i, j)] = (dist, costo)

        for i in nodos:
            for j in nodos:
                if i < j and (i, j) not in aristas:
                    aristas[(i, j)] = (VALOR_GRANDE, VALOR_GRANDE)

        for (i, j), (d, c) in aristas.items():
            f.write(f"{i} {j} {d} {c}\n")

    print(f"Instancia generada y guardada en {nombre_archivo}")



def generar_coords_uniforme(cant_clientes, rango_coord):
    coords = {i: (random.uniform(0, rango_coord), random.uniform(0, rango_coord)) for i in range(1, cant_clientes + 1)}
    coords[0] = (rango_coord / 2, rango_coord / 2)  # depósito en el centro
    return coords

def generar_coords_cluster(cant_clientes, rango_coord, clusters=3, std=0.1):
    centros = [(random.uniform(0.2 * rango_coord, 0.8 * rango_coord),
                random.uniform(0.2 * rango_coord, 0.8 * rango_coord)) for _ in range(clusters)]
    coords = {}
    for i in range(1, cant_clientes + 1):
        cx, cy = random.choice(centros)
        x = np.clip(random.gauss(cx, std * rango_coord), 0, rango_coord)
        y = np.clip(random.gauss(cy, std * rango_coord), 0, rango_coord)
        coords[i] = (x, y)
    coords[0] = (rango_coord / 2, rango_coord / 2)
    return coords

def generar_coords_anillo(cant_clientes, rango_coord, prop_nucleo=0.2, radio_interno=0.3, radio_externo=0.5):
    centro = (rango_coord / 2, rango_coord / 2)
    n_nucleo = int(cant_clientes * prop_nucleo)
    n_anillo = cant_clientes - n_nucleo
    coords = {}

    # Núcleo
    for i in range(1, n_nucleo + 1):
        x = np.clip(random.gauss(centro[0], 0.05 * rango_coord), 0, rango_coord)
        y = np.clip(random.gauss(centro[1], 0.05 * rango_coord), 0, rango_coord)
        coords[i] = (x, y)

    # Anillo
    for j in range(1, n_anillo + 1):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(radio_interno * rango_coord, radio_externo * rango_coord)
        x = centro[0] + radius * math.cos(angle)
        y = centro[1] + radius * math.sin(angle)
        coords[n_nucleo + j] = (np.clip(x, 0, rango_coord), np.clip(y, 0, rango_coord))

    coords[0] = centro
    return coords
# --------------------------
# 2. Leer instancia y grafo
# --------------------------

def leer_instancia(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        lineas = f.readlines()

    cant_clientes = int(lineas[0])
    cant_refrigerados = int(lineas[3])
    cant_exclusivos = int(lineas[4 + cant_refrigerados])
    header = lineas[:5 + cant_refrigerados + cant_exclusivos]
    aristas = lineas[len(header):]

    grafo = defaultdict(list)
    for linea in aristas:
        i, j, d, c = map(int, linea.strip().split())
        grafo[i].append((j, d, c))
        grafo[j].append((i, d, c))
    return grafo, header

# --------------------------
# 3. Dijkstra genérico
# --------------------------

def dijkstra(grafo, origen, n, modo='dist'):
    peso = 1 if modo == 'dist' else 2
    distancias = [float('inf')] * n
    distancias[origen] = 0
    heap = [(0, origen)]

    while heap:
        d_actual, nodo = heapq.heappop(heap)
        if d_actual > distancias[nodo]:
            continue
        for vecino, d, c in grafo[nodo]:
            peso_actual = d if peso == 1 else c
            nueva_d = d_actual + peso_actual
            if nueva_d < distancias[vecino]:
                distancias[vecino] = nueva_d
                heapq.heappush(heap, (nueva_d, vecino))
    return distancias

# --------------------------
# 4. Actualizar distancias mínimas
# --------------------------

def actualizar_distancias_minimas(nombre_archivo, salida="Instancia_Dist_Mod.txt"):
    grafo, header = leer_instancia(nombre_archivo)
    nodos = list(grafo.keys())
    n = max(nodos) + 1
    resultado = {}

    for i in nodos:
        dist = dijkstra(grafo, i, n, modo='dist')
        for j in nodos:
            if i != j:
                par = tuple(sorted((i, j)))
                if par not in resultado or dist[j] < resultado[par][0]:
                    resultado[par] = [dist[j], VALOR_GRANDE]

    for i in nodos:
        for j, _, c in grafo[i]:
            par = tuple(sorted((i, j)))
            if par in resultado:
                resultado[par][1] = c

    with open(salida, 'w') as f:
        f.writelines(header)
        for (i, j), (d, c) in sorted(resultado.items()):
            f.write(f"{i} {j} {d} {c}\n")
    print(f"Distancias mínimas actualizadas en: {salida}")

# --------------------------
# 5. Actualizar costos mínimos
# --------------------------

def actualizar_costos_minimos(nombre_archivo, salida="Instancia_F.txt"):
    grafo, header = leer_instancia(nombre_archivo)
    nodos = list(grafo.keys())
    n = max(nodos) + 1
    resultado = {}

    for i in nodos:
        costos = dijkstra(grafo, i, n, modo='costo')
        for j in nodos:
            if i != j:
                par = tuple(sorted((i, j)))
                if par not in resultado or costos[j] < resultado[par][1]:
                    resultado[par] = [VALOR_GRANDE, costos[j]]

    for i in nodos:
        for j, d, _ in grafo[i]:
            par = tuple(sorted((i, j)))
            if par in resultado:
                resultado[par][0] = d

    with open(salida, 'w') as f:
        f.writelines(header)
        for (i, j), (d, c) in sorted(resultado.items()):
            f.write(f"{i} {j} {d} {c}\n")
    print(f"Costos mínimos actualizados en: {salida}")

# --------------------------
# 6. Ejemplo de uso
# --------------------------

if __name__ == "__main__":
    generar_instancia("Instancia_200_Cluster.txt", cant_clientes=200, costo_repartidor=50, dist_max=150,
                      cant_refrigerados=40, cant_exclusivos=40, porcentaje_conectividad=0.5, rango_coord=1000, tipo_distribucion='clusters')
    
    actualizar_distancias_minimas("Instancia_200_Cluster.txt", "Instancia_200_Cluster.txt")
    actualizar_costos_minimos("Instancia_200_Cluster.txt", "Instancia_200_Cluster.txt")
    
# cant_clientes = 200
# costo_repartidor = 50
# dist_max = 150
# cant_refrigerados = 40 (20%)
# cant_exclusivos = 40 (20%)
# Distribucion Clusters de 3