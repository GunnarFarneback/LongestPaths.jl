# LongestPaths.jl

LongestPaths is a Julia package dedicated to finding long simple paths
or cycles, i.e. no repeated vertex, in a graph, as well as upper
bounds on the maximum length.

The longest path problem is NP-hard, so the time needed to find the
solution grows quickly with the size of the graph, unless it has some
advantageous structure.

At this time two search functions are provided:

    find_longest_path(graph, first_vertex = 1, last_vertex = 0; kwargs)

Find the longest simple path in a directed Graphs.jl `graph`, starting
with `first_vertex` and ending in `last_vertex`. If `last_vertex` is 0
the path may end anywhere. If time limits or other restrictions
prevent finding an optimal path, an upper bound on the maximum length
is returned together with the longest path found.

    find_longest_cycle(graph, first_vertex = 0; kwargs)

Find the longest simple cycle in a directed Graphs.jl `graph`, which
includes `first_vertex`. If `first_vertex` is 0 the cycle may be
anywhere. If time limits or other restrictions prevent finding an
optimal cycle, an upper bound on the maximum length is returned
together with the longest cycle found.

## Adding LongestPaths

LongestPaths is a registered Julia package.

In Julia `pkg` mode (press `]`):
```
pkg> add LongestPaths
```

## Usage Example
```
julia> using LongestPaths, Graphs

julia> g = erdos_renyi(500, 0.005, is_directed=true, seed=13)
{500, 1286} directed simple Int64 graph

julia> find_longest_path(g)
  1     1 [267 352] 0 0 Optimal 352.0 352.0
  2     2 [267 352] 0 24 Optimal 352.0 352.0
  3     2 [267 352] 0 112 Optimal 352.0 352.0
  4     2 [338 352] 0 132 Optimal 352.0 352.0
  5     2 [338 352] 0 146 Optimal 352.0 352.0
  6     3 [352 352] 0 159 Optimal 352.0 352.0
Longest path with bounds [352, 352] and a recorded path of length 352.
```
For large problems you most likely want to add some restriction on how
long the search can go on. See the doc string.

## Weighted Longest Paths

Edge-weighted longest path problems are supported. Example, continued
from above:
```
julia> w = Dict((v1, v2) => mod1(v1 + v2, 2) for (v1, v2) in Tuple.(edges(g)));

julia> find_longest_path(g, weights = w)
  1     0 [349.0 539.0] 0.0 0 Optimal 539.0 539.0
  2     0 [413.0 539.0] 0.0 54 Optimal 539.0 539.0
  3     0 [457.0 539.0] 0.0 128 Optimal 539.0 539.0
  4     0 [539.0 539.0] 0.0 139 Optimal 539.0 539.0
Longest path with bounds [539.0, 539.0] and a recorded path of weight 539.
```

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

(These constraints are for searching for a path starting in a
specified vertex and ending anywhere. The other search variants have
slightly different constraints, documented in the source code.)

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

* Add more tests.

* Add benchmarks.

* Convert from MathProgBase to MathOptInterface.

* Generalize to other solvers than Cbc. This should be more or less
  straightforward, but is at the moment hindered by an absence of
  licenses for commercial solvers like Gurobi or CPLEX.

* Add search methods for finding long paths that don't involve solving
  optimization problems. These are already developed but need to be
  upgraded to Julia 1.0, integrated with Graphs.jl, and polished.

* General polishing of the code.
