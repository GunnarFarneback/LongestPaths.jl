# LongestPaths.jl

The LongestPaths package is dedicated to finding long simple paths,
i.e. no repeated vertex, in a graph, as well as upper bounds on the
maximum length.

The longest path problem is NP-hard, so the time needed to find the
solution grows quickly with the size of the graph, unless it has some
advantageous structure.

At this time only one function is provided:

    longest_path(graph; kwargs)

Find the longest simple path in a directed LightGraphs graph, starting
with vertex one and ending anywhere. If time limits or other
restrictions prevent finding an optimal path, an upper bound on the
maximum length is returned together with the longest path found.


## Theory

Although developed independently, the main ideas used here coincide
with

Leonardo Taccari. Integer programming formulations for the elementary
shortest path problem. *European Journal of Operational Research*,
252(1):122–130, 2016.

See the reference for a rigid motivation and further references to
similar approaches for related problems such as the travelling
salesman problem.

In short, the problem is posed as an Integer Linear Program with
binary variables. There is one variable for each edge with 1 meaning
that the edge is included in the path. The constraints are, with the
convention that sum of edges means the sum of the corresponding
variables:

* The sum of incoming edges to the first vertex is 0.

* The sum of incoming edges to all other vertices must be between 0
  and 1.

* The sum of outgoing edges from a vertex minus the sum of the
  incoming edges is between 0 and 1 for the first vertex.

* The sum of outgoing edges from a vertex minus the sum of the
  incoming edges is between -1 and 0 for all other vertices

The objective function is the sum of all edge variables, which is
maximized.

Clearly all simple paths starting from the first vertex are feasible
solutions to this problem, so any upper bound of the optimization
problem is an upper bound to the length of the simple paths. Moreover,
any upper bound to the LP relaxation of the problem (integer
constraints ignored) is also an upper bound of the path length.

Unfortunately these constraints are not sufficient to only allow
simple paths. Additional feasible solutions consist of one path
complemented with an arbitrary number of *cycles*. Cycles can be
eliminated by adding constraints, e.g. that the sum of the edge
variables in the cycle must be at most `n - 1` for a cycle of length
`n`. If such a constraint is added for every possible cycle in the
graph, the optimization would only have the simple paths as feasible
solutions and the optimal solution would give a maximum length path.
However, the number of possible cycles grows very quickly with the
graph size and the number of constraints would soon become
intractable. Instead we only add constraints for the cycles that we
actually find in the solutions and then iterate, with the hope of
reaching an optimal path long before every cycle has been added to the
constraints.

Note: instead of limiting the cycle length, the "generalized cutset
inequalities" from Taccari are a more efficient way to constrain
cycles, in particular since they have a greater effect on the LP
relaxation. That is also the default in the `longest_path` function.

## Future Plans

* Generalize to a user selected starting vertex, a fixed ending
  vertex, and to finding longest cycles. This is all straightforward.

* Generalize to weighted longest paths. This is mostly
  straightforward.

* Add more tests and set up CI.

* Add benchmarks.

* Possibly convert from MathProgBase to MathOptInterface.

* Generalize to other solvers than Cbc. This should be more or less
  straightforward, but is at the moment hindered by an absence of
  licenses for commercial solvers like Gurobi or CPLEX.

* Add search methods for finding long paths that don't involve solving
  optimization problems. These are already developed but need to be
  upgraded to Julia 1.0, integrated with LightGraphs, and polished.

* General polishing of the code.
