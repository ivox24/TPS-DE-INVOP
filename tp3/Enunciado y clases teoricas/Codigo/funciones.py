import numpy as np

# -------------------------------
# Funcion Para Crear Instancias
# -------------------------------

def crear_instancia(nombre_archivo="instancia.txt", n_dim=50, n_puntos=1000,
                    rango_coords=(-500, 500), rango_pesos=(1, 1), distribucion="uniforme", seed=None):

    if seed is not None:
        np.random.seed(seed)

    if distribucion == "aleatorio":
        puntos = np.random.uniform(rango_coords[0], rango_coords[1], size=(n_puntos, n_dim))
        
    elif distribucion == "uniforme":
        
        puntos = np.linspace(rango_coords[0], rango_coords[1], n_puntos * n_dim)
        puntos = puntos.reshape(n_puntos, n_dim)
        
    elif distribucion == "clusters":
        n_clusters = 4
        puntos_por_cluster = n_puntos // n_clusters
        extra = n_puntos % n_clusters

        puntos = []
        for i in range(n_clusters):
            centro = np.random.uniform(rango_coords[0], rango_coords[1], size=(n_dim,))
            cantidad = puntos_por_cluster + (1 if i < extra else 0)
            dispersion = (rango_coords[1] - rango_coords[0]) * 0.05
            cluster = np.random.normal(loc=centro, scale=dispersion, size=(cantidad, n_dim))
            puntos.append(cluster)

        puntos = np.vstack(puntos)
        
    else:
        raise ValueError("Distribución no válida. Usar 'aleatorio', 'uniforme' o 'clusters'.")
    
    
    pesos = np.random.uniform(rango_pesos[0], rango_pesos[1], size=n_puntos).reshape(-1, 1)
    
    datos = np.hstack((puntos, pesos))

    with open(nombre_archivo, "w") as f:
        f.write("# Formato: x1 x2 ... xn w\n")
        for fila in datos:
            f.write(" ".join(f"{valor:.6f}" for valor in fila) + "\n")

    print(f"✅ Instancia guardada en '{nombre_archivo}'")
    return datos

# -------------------------------
# Funcion Para Leer Instancias
# -------------------------------

def leer_instancia(nombre_archivo):
    
    datos = []

    with open(nombre_archivo, "r") as f:
        for linea in f:
            if linea.strip().startswith("#") or not linea.strip():
                continue  # Ignorar comentarios y líneas vacías
            valores = list(map(float, linea.strip().split()))
            datos.append(valores)

    datos = np.array(datos)
    puntos = datos[:, :-1]  # todas las columnas menos la última
    pesos = datos[:, -1]    # última columna

    return puntos, pesos