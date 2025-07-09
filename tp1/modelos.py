import cplex

problem = cplex.Cplex()
problem.set_problem_type(cplex.Cplex.problem_type.LP)
problem.objective.set_sense(problem.objective.sense.maximize)

# Variables
problem.variables.add(names=["x1", "x2"], lb=[0, 0])

# Función objetivo: 5.900 x1 + 2900 x2 + 1200 x3 
problem.objective.set_linear([("x1", 7), ("x2", 6)])


# Restricciones
problem.linear_constraints.add(
     lin_expr=[
        [["x1", "x2"], [1, 3]],
        [["x1", "x2"], [2, 1]],
        [["x1", "x2"], [2, 3]]
        
     ],
     senses=["L", "L", "L"],
     rhs=[6, 9, 10]
 )

# Resolver
problem.solve()

print("x1 =", problem.solution.get_values("x1"))
print("x2 =", problem.solution.get_values("x2"))
print("Z =", problem.solution.get_objective_value())
