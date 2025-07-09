#Modelo 2.1
#Modelo con la nueva metodología (camiones y repartidores), sin restricciones de exclusividad ni de 4 repartos mínimos por repartidor.

import sys
import cplex

TOLERANCE = 1e-6

class InstanciaRecorridoMixto:
    def __init__(self):
        self.cantidad_clientes = 0
        self.costo_repartidor = 0
        self.d_max = 0
        self.refrigerados = []
        self.exclusivos = []
        self.distancias = []        
        self.costos = []
        self.a_ij = dict()  #diccionario booleano
        

    def leer_datos(self, filename):
        f = open(filename)

        self.cantidad_clientes = int(f.readline())
        self.costo_repartidor = int(f.readline())
        self.d_max = int(f.readline())

        self.distancias = [[1e6 for _ in range(self.cantidad_clientes + 1)] for _ in range(self.cantidad_clientes + 1)]
        self.costos = [[1e6 for _ in range(self.cantidad_clientes + 1)] for _ in range(self.cantidad_clientes + 1)]

        cantidad_refrigerados = int(f.readline())
        for _ in range(cantidad_refrigerados):
            self.refrigerados.append(int(f.readline()))

        cantidad_exclusivos = int(f.readline())
        for _ in range(cantidad_exclusivos):
            self.exclusivos.append(int(f.readline()))

        for linea in f.readlines():
            row = list(map(float, linea.strip().split()))
            i, j = int(row[0]), int(row[1])
            dij, cij = float(row[2]), float(row[3])
            self.distancias[i][j] = dij
            self.distancias[j][i] = dij
            self.costos[i][j] = cij
            self.costos[j][i] = cij

        #diccionario booleano aij
        for i in range(len(self.distancias)):
            for j in range(len(self.distancias)):
                if i != j:
                    if self.distancias[i][j] <= self.d_max:
                        self.a_ij[(i, j)] = 1
                    else:
                        self.a_ij[(i, j)] = 0

        f.close()

def cargar_instancia():
    nombre_archivo = "Instancia_4.txt"
    instancia = InstanciaRecorridoMixto()
    instancia.leer_datos(nombre_archivo)
    return instancia

def agregar_variables(prob, instancia):
    nombres = []
    obj = []
    tipos = []
    lb = []
    ub = []

    n = instancia.cantidad_clientes
    N = list(range(n + 1))

    # Variables x_ij y z_ij
    for i in N:
        for j in N:
            if i != j:
                nombres.append(f"x_{i}_{j}")
                obj.append(instancia.costos[i][j])
                tipos.append("B")
                lb.append(0)
                ub.append(1)

                # solo se creen las variables z_ij que están permitidas (según aij).
                if instancia.a_ij.get((i, j), 0) == 1:
                    nombres.append(f"z_{i}_{j}")
                    obj.append(instancia.costo_repartidor)
                    tipos.append("B")
                    lb.append(0)
                    ub.append(1)

    for i in N:
        nombres.append(f"w_{i}")
        obj.append(0)
        tipos.append("B")
        lb.append(0)
        ub.append(1)

    for i in range(n + 1):
        nombres.append(f"u_{i}")
        obj.append(0)
        tipos.append("I")
        lb.append(0 if i == 0 else 1)
        ub.append(0 if i == 0 else n)

    prob.variables.add(obj=obj, lb=lb, ub=ub, types=tipos, names=nombres)


def agregar_restricciones(prob, instancia):
    n = instancia.cantidad_clientes
    N = list(range(n + 1))
    M1 = n
    M2 = instancia.d_max + 1

    def var(name):
        return prob.variables.get_indices(name)
   #1
    for i in N:
        z_ij = [var(f"z_{i}_{j}") for j in N if i != j and instancia.a_ij.get((i, j), 0) == 1]
        if z_ij:
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=z_ij + [var(f"w_{i}")], val=[1]*len(z_ij) + [-(n - 1)])],
                senses=["L"], rhs=[0], names=[f"max_reparto_{i}"]
            )
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=z_ij + [var(f"w_{i}")], val=[1]*len(z_ij) + [-1])],
                senses=["G"], rhs=[0], names=[f"min_reparto_{i}"]
            )

    #2
    for i in N:
        for j in N:
            if i != j and instancia.a_ij.get((i, j), 0) == 1:
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[var(f"z_{i}_{j}")], val=[1])],
                    senses=["L"],
                    rhs=[instancia.a_ij[(i, j)]], #Este a_ij está precalculado.
                    #Sin esto, declarando a a_ij como variable, el modelo sería más lento, porque tendría que eliminar muchas más columnas inncesarias.
                    names=[f"z_leq_a_{i}_{j}"]
                )

    #3
    for i in N:
        z_ij = [var(f"z_{i}_{j}") for j in instancia.refrigerados if i != j and instancia.a_ij.get((i, j), 0) == 1]
        if z_ij:
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=z_ij, val=[1]*len(z_ij))],
                senses=["L"], rhs=[1], names=[f"refrigerado_{i}"]
            )

    #4
    for j in N:
        if j != 0:  # excluye el depósito
            x_ij = [var(f"x_{i}_{j}") for i in N if i != j]
            z_ij = [var(f"z_{i}_{j}") for i in N if i != j and instancia.a_ij.get((i, j), 0) == 1]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=x_ij + z_ij, val=[1] * (len(x_ij) + len(z_ij)))],
                senses=["E"],
                rhs=[1],
                names=[f"visita_unica_{j}"]
            )

    #5
    for i in range(1, n + 1):
        x_ij = [var(f"x_{i}_{j}") for j in N if j != i]
        x_ji = [var(f"x_{j}_{i}") for j in N if j != i]
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=x_ij + x_ji, val=[1]*len(x_ij) + [-1]*len(x_ji))],
            senses=["E"], rhs=[0], names=[f"flujo_{i}"]
        )

    #6
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[var(f"x_0_{j}") for j in range(1, n + 1)], val=[1]*n)],
        senses=["E"], rhs=[1], names=["salida_deposito"]
    )
    #7
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[var(f"x_{i}_0") for i in range(1, n + 1)], val=[1]*n)],
        senses=["E"], rhs=[1], names=["entrada_deposito"]
    )

    #8
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[var(f"u_{i}"), var(f"u_{j}"), var(f"x_{i}_{j}")],
                        val=[1, -1, n]
                    )],
                    senses=["L"],
                    rhs=[n-1],
                    names=[f"mtz_{i}_{j}"]
                )

    #9
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[var("u_0")], val=[1])],
        senses=["E"], rhs=[0], names=["u0_cero"]
    )

    #10
    for i in range(1, n + 1):
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[var(f"u_{i}")], val=[1])],
            senses=["G"], rhs=[1], names=[f"u_min_{i}"]
        )
        prob.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=[var(f"u_{i}")], val=[1])],
            senses=["L"], rhs=[n], names=[f"u_max_{i}"]
        )

    #11
    prob.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=[var("w_0")], val=[1])],
        senses=["E"],
        rhs=[0],
        names=["w0_igual_cero"]
    )

    #12
    for i in N:
        if instancia.a_ij.get((i, 0), 0) == 1:
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[var(f"z_{i}_0")], val=[1])],
                senses=["E"],
                rhs=[0],
                names=[f"z_{i}_0_igual_cero"]
            )

    #13
    for i in N:
        for j in N:
            if i != j:
                if instancia.a_ij.get((i, j), 0) == 1 and instancia.a_ij.get((j, i), 0) == 1:
                    prob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[var(f"z_{i}_{j}"), var(f"z_{j}_{i}")],
                            val=[1, 1]
                        )],
                        senses=["L"],
                        rhs=[1],
                        names=[f"anti_ciclo_repartidor_{i}_{j}"]
                    )

    #14
    for i in N:
        z_ij = [var(f"z_{i}_{j}") for j in N if i != j and instancia.a_ij.get((i, j), 0) == 1]
        x_ij = [var(f"x_{i}_{j}") for j in N if i != j]
        if z_ij and x_ij:
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=z_ij + x_ij, val=[1]*len(z_ij) + [-(n - 1)]*len(x_ij))],
                senses=["L"],
                rhs=[0],
                names=[f"reparto_condicional_{i}"]
            )


def armar_lp(prob, instancia):
    agregar_variables(prob, instancia)
    agregar_restricciones(prob, instancia)
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.write("modelo_camion_y_repartidores.lp")

def resolver_lp(prob):
    prob.parameters.timelimit.set(900)
    prob.solve()

def mostrar_solucion(prob, instancia):
    status = prob.solution.get_status_string()
    valor_obj = prob.solution.get_objective_value()
    print(f"Función objetivo: {valor_obj} ({status})")

    valores = prob.solution.get_values()
    nombres = prob.variables.get_names()

    print("\nVariables con valor positivo:")
    for nombre, valor in zip(nombres, valores):
        if valor > TOLERANCE:
            print(f"{nombre} = {round(valor, 6)}")  #Modificado para evitar casos de 1.0000000000000002 por ejemplo


def main():
    instancia = cargar_instancia()
    prob = cplex.Cplex()
    #--- BÚSQUEDA ----------------------------------------------------------
    prob.parameters.mip.strategy.search.set(1)         # 1 = depth-first (menos RAM)
    
    #--- HEURÍSTICAS -------------------------------------------------------
    #prob.parameters.heuristic.set(20)                # 3 % de tiempo raíz a heurísticas
    prob.parameters.mip.strategy.heuristicfreq.set(20) # cada 20 nodos internos

    #--- CORTES ------------------------------------------------------------
    prob.parameters.mip.cuts.gomory.set(-1)            # sin cortes Gomory
    prob.parameters.mip.cuts.mircut.set(-1)            # sin MIR

    armar_lp(prob, instancia)
    resolver_lp(prob)
    mostrar_solucion(prob, instancia)
    print_camion_route(prob, instancia)
    print_repartidores_rutas(prob, instancia)
    
def print_camion_route(prob, instancia):
    n = instancia.cantidad_clientes
    edges = [(i, j) for i in range(n + 1) for j in range(n + 1)
             if i != j and prob.solution.get_values(f"x_{i}_{j}") > 0.5]
    succ = {i: j for i, j in edges}
    camino = []
    nodo = 0
    while True:
        camino.append(nodo)
        nodo = succ.get(nodo, None)
        if nodo is None or nodo == 0:
            camino.append(0)
            break
    print("Ruta camión:", " → ".join(map(str, camino)))

def print_repartidores_rutas(prob, instancia):
    print("\nRutas de repartidores (z_ij = 1):")
    valores = prob.solution.get_values()
    nombres = prob.variables.get_names()

    for nombre, valor in zip(nombres, valores):
        if nombre.startswith("z_") and valor > 1e-6:
            i, j = map(int, nombre[2:].split("_"))
            print(f"Repartidor: {i} → {j}")

if __name__ == '__main__':
    main()
