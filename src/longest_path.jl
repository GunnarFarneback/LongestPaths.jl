export LongestPathOrCycle, find_longest_path, find_longest_cycle,
       is_path, is_cycle, is_simple_path, is_simple_cycle

using MathProgBase
using MathProgBase.SolverInterface
using Clp
using Cbc
using LightGraphs
using SparseArrays
using Printf
using Random

abstract type AbstractWeightedPath{T} end
struct UnweightedPath <: AbstractWeightedPath{Int} end
struct UnweightedCycle <: AbstractWeightedPath{Int} end
struct WeightedPath{T} <: AbstractWeightedPath{T}
    weights::Dict{Tuple{Int, Int}, T}
end
struct WeightedCycle{T} <: AbstractWeightedPath{T}
    weights::Dict{Tuple{Int, Int}, T}
end

are_cycle_weights(::UnweightedPath) = false
are_cycle_weights(::UnweightedCycle) = true
are_cycle_weights(::WeightedPath) = false
are_cycle_weights(::WeightedCycle) = true

are_weighted(::UnweightedPath) = false
are_weighted(::UnweightedCycle) = false
are_weighted(::WeightedPath) = true
are_weighted(::WeightedCycle) = true

# Turn unweighted weights into explicit weights with value 1.
explicit_weights(w::WeightedPath, graph) = w
explicit_weights(w::WeightedCycle, graph) = w
explicit_weights(w::UnweightedPath, graph) = WeightedPath(unit_weights(graph))
explicit_weights(w::UnweightedCycle, graph) = WeightedCycle(unit_weights(graph))
unit_weights(graph) = Dict(Tuple(edge) => 1 for edge in edges(graph))

are_integer_weights(::AbstractWeightedPath{<:Integer}) = true
are_integer_weights(::Any) = false

"""
    LongestPathOrCycle

Type used for the return values of `longest_path`. See the function
documentation for more information.
"""
mutable struct LongestPathOrCycle{T}
    weights::AbstractWeightedPath
    lower_bound::T
    upper_bound::T
    longest_path::Vector{Int}
    internals::Dict{String, Any}
end

# TODO: Handle empty paths nicely.
function Base.show(io::IO, x::LongestPathOrCycle)
    kind = are_cycle_weights(x.weights) ? "cycle" : "path"
    if isempty(x.longest_path)
        println(io, "Longest $(kind) problem without a feasible solution.")
    else
        n = path_length(x.longest_path, x.weights)
        length_str = are_weighted(x.weights) ? "weight" : "length"
        println(io, "Longest $(kind) with bounds [$(x.lower_bound), $(x.upper_bound)] and a recorded $(kind) of $(length_str) $(n).")
    end
end

main_docstring = """
    find_longest_path(graph, first_vertex = 1, last_vertex = 0)

Find the longest simple path in `graph` starting from `first_vertex`
and ending in `last_vertex`. No vertex may be visited more than once.
If `last_vertex` is 0, the path may end anywhere. `first_vertex`
cannot be 0.

    find_longest_cycle(graph, first_vertex = 0)

Find the longest simple cycle in `graph` that includes `first_vertex`.
No vertex may be visited more than once. If `first_vertex` is 0, the
cycle can be anywhere.

Given sufficient time and memory, these will succeed in finding the
longest path or cycle, but since finding the longest path/cycle is an
NP-hard problem, the required time will grow quickly with the graph
size.

For the time being, `graph` must be a **directed** graph from the
`LightGraphs` package. The algorithm works for undirected graphs if
they are represented as directed graphs with pairs of edges in both
directions.

    find_longest_path(...; kwargs)
    find_longest_cycle(...; kwargs)

By adding keyword arguments it is possible to guide the search or
obtain non-optimal solutions and upper bounds in shorter time than the
full solution.

Note: Unless otherwise specified, both paths and cycles are called
"paths" in the following and they are always assumed to be simple,
i.e. no repeated vertices are allowed.

*Keyword arguments:*

* `weights`: Edge weights for weighted longest path searches. If
  omitted all weights are implicitly one. Specified as a dictionary
  mapping tuples to numbers. `(v1, v2) => w` means that the edge from
  `v1` to `v2` has the weight `w`.

* `initial_path`: Search can be warmstarted by providing a path as a
  vector of vertices. If the provided vertices do not follow a valid
  simple path, they are ignored.

* `lower_bound`: User provided lower bound. Search will stop when the
  upper bound reaches `lower_bound`, even if no path of that length
  has been found. The provided `lower_bound` will be ignored if a
  stronger bound is found during search.

* `upper_bound`: User provided upper bound. Search will stop when the
  lower bound reaches `upper_bound`, even if a longer path exists.
  The provided `upper_bound` will be ignored if a stronger bound is
  found during search.

* `solver_mode`: Search can be made in three different modes, `"lp"`,
  `"ip"`, or `"lp+ip"`. In the default `"ip"` mode, integer programs
  are solved. This can be slow but is guaranteed to eventually find
  the optimal solution. In the `"lp"` mode only the linear program
  relaxations are solved. This is unlikely to find a long path but
  will quickly find good upper bounds. The final `"lp+ip"` mode
  alternates between `"lp"` and `"ip"` mode and is sometimes more
  efficient than `"ip"` only to find the optimal solution.

* `cycle_constraint_mode`: During the search, cycles that are
  unconnected to the path will be found and iteratively eliminated by
  adding constraints to the integer programs or the linear program
  relaxations. Different kinds of constraints can be added for this
  purpose and `cycle_constraint_mode` can be set to `"cutset"`,
  `"cycle"` or `"both"`. It is likely that the default `"cutset"`
  option is the best choice.

* `initial_cycle_constraints`: By setting this larger than the default
  value 0, all cycles in the graph of length up to this number will be
  eliminated from the search before starting. This gives a tradeoff
  between the number of iterations needed and the time required for
  each iteration. It probably does not pay off to set this higher than
  about 3, and possibly 0 is the best setting.

* `max_iterations`: Stop search after this number of iterations.
  Defaults to a very high number.

* `time_limit`: Stop search after this number of seconds. Defaults to
  a very high number.

* `solver_time_limit`: Maximum time to spend in each iteration to
  solve integer programs. This will be gradually increased if the IP
  solver does not find a useful solution in the allowed time. LP
  solutions are not affected by this option and are always solved to
  optimality. Defaults to 10 seconds.

* `max_gap`: Allowed gap for IP solutions. This will automatically be
  reduced to be smaller than the distance between the lower and upper
  bounds. Default is 0. Higher values can speed up the early
  iterations in `"ip"` mode.

* `use_ip_warmstart`: Use warmstart when solving integer programs.
  Normally this speeds up the solution substantially but it also
  causes the Cbc solver to emit trace outputs. Default is true.

* `log_level`: Amount of verbosity during search. 0 is maximally
  quiet. The default value 1 only prints progress for each iteration.
  Higher values add diagnostics from the IP solver calls.

* `new_longest_path_callback`: A function provided here will be called
  every time a new longest path has been found. This gives an
  opportunity to save partial solutions of very long running searches.
  The function should accept one argument, which is the longest path
  expressed as a vector of vertices. Default is a function that does
  nothing. The return value from the function is ignored.

* `iteration_callback`: A function provided here is called during each
  iteration. The function should take one argument which is a named
  tuple with diagnostic information, see the code for exact
  specifications. Return `true` to continue search and `false` to stop
  search. The default is the `print_iteration_data` function.

* `preprocess`: Whether to preprocess the graph before starting
  the search. Defaults to true. This performs the following steps:
  1. Convert undirected graphs to directed graphs.
  2. Remove self-loops.
  3. Remove vertices and edges which can provably not be part of an
     optimal solution.
  The main reason to disable this are to save time and/or memory if
  you know that the preprocessing will not be helpful. The
  `iteration_callback` and the internals of the returned result will
  relate to the preprocessed graph if `preprocess` is enabled, but
  vertices can be mapped back to the original graph via the provided
  `vertex_mapping`.

* `reduce_unbranched`: Whether to optimize the graph such that
  vertices with two neighbors are eliminated where possible. Defaults
  to false. This turns unweighted searches into weighted searches. The
  same caveats with respect to introspection as `preprocess` apply.

The return value is of the `LongestPathOrCycle` type and contains the
following fields:

* `is_cycle`: Whether the search produced a cycle or a path.

* `lower_bound`: Lower bound for the length of the longest path.

* `upper_bound`: Upper bound for the length of the longest path.

* `longest_path`: Vector of the vertices in the longest found path.

* `internals`: Dict containing a variety of information about the search.

Notes:

* Path lengths are reported by number of edges, not number of
  vertices. For cycles these are the same, for paths the number of
  edges is one less than the number of vertices. For weighted searches
  the length is the sum of the weights along the path.

* If there is no outgoing edge from `first_vertex` in a path search
  going anywhere, the length is reported as 0 and the returned
  `longest_path` contains the single vertex `first_vertex`. This is
  also the case if `first_vertex == last_vertex`.

* In other searches, if there exists no path or cycle matching the
  specifications, the length is reported as 0 and the returned
  `longest_path` is empty.

* In weighted searches, if the weights are of an `Integer` type, the
  search knows that the optimum is an integer value and can round
  bounds more aggressively.

* Weights may be zero or negative. If all weights are non-positive the
  problem effectively becomes a shortest path problem, which is fine
  but can be solved considerably more efficiently with other methods.
"""

"$(main_docstring)"
function find_longest_path(graph, first_vertex::Integer = 1,
                           last_vertex::Integer = 0;
                           weights = nothing, kwargs...)
    w = get_weights(weights, false)

    # TODO: Do the reversal ourselves and perform the search?
    if first_vertex == 0 && last_vertex != 0
        error("Search from anywhere to a specified vertex is not supported. Reverse the graph and search in the opposite direction instead.")
    end

    if first_vertex == last_vertex == 0
        error("Search cannot be done for a path starting and ending anywhere.")
    end

    if first_vertex == last_vertex != 0
        path = [first_vertex]
        lb = ub = path_length(path, w)
        return LongestPathOrCycle(w, lb, ub, path, Dict{String, Any}())
    end

    # TODO: Check why this really is necessary. (I.e. why the main
    # function doesn't handle this case correctly.)
    if last_vertex == 0 && isempty(outneighbors(graph, first_vertex))
        path = [first_vertex]
        lb = ub = path_length(path, w)
        return LongestPathOrCycle(w, lb, ub, path, Dict{String, Any}())
    end

    return _pre_find_longest_path(graph, w, first_vertex, last_vertex; kwargs...)
end

"$(main_docstring)"
function find_longest_cycle(graph, first_vertex = 0;
                            weights = nothing, kwargs...)
    w = get_weights(weights, true)
    return _pre_find_longest_path(graph, w, first_vertex, first_vertex; kwargs...)
end

function get_weights(weights::Nothing, is_cycle)
    return is_cycle ? UnweightedCycle() : UnweightedPath()
end

function get_weights(weights::Dict{Tuple{Int, Int}, <:Any}, is_cycle)
    return is_cycle ? WeightedCycle(weights) : WeightedPath(weights)
end

# Intermediate landing point to possibly perform preprocessing of the
# graph before continuing to the main search function.
#
# 1. Convert an undirected graph to a directed graph.
# 2. Remove self loops if any.
# 3. Remove unneeded vertices.
#   a) If search for path ending anywhere:
#      Only keep vertices which can be reached by a path from the
#      first vertex.
#   b) If search for path ending in a specific vertex:
#      Only keep vertices which can be reached by a path from the
#      first vertex and reach the last vertex by a path.
#   c) If searching for a loop through a given vertex:
#      Only keep the strongly connected component containing this
#      vertex.
#   d) If searching for a loop anywhere:
#      Remove strongly connected components of size one.
#      TODO: This case can be improved further by running separate
#      searches for each strongly connected component, preferably with
#      good ordering heuristics so some searches can be skipped
#      entirely.
# 4. Reduce unbranched sequences of edges to one edge weighted as the
#    sum of the weights of the removed edges.
#
# 1-3 are applied if `preprocess` is true and 4 if `reduce_unbranched`
# is true.
function _pre_find_longest_path(graph, weights::AbstractWeightedPath,
                                first_vertex, last_vertex;
                                initial_path = Int[],
                                preprocess = true,
                                reduce_unbranched = false,
                                kwargs...)
    input_graph = graph
    vertex_mapping = 1:nv(graph)
    mapped_weights = weights
    if preprocess
        # 1. Convert to directed graph if it isn't already.
        if !is_directed(graph)
            # TODO: Support also this case. Duplicate the weights to
            # both directions.
            if are_weighted(weights)
                error("Weighted undirected graphs are not supported at this time. Convert your undirected graph to a directed representation.")
            end
            graph = SimpleDiGraph(graph)
        end
        # 2. Remove self loops if there are any.
        if has_self_loops(graph)
            if graph === input_graph
                graph = copy(graph)
            end
            for v in vertices(graph)
                if has_edge(graph, v, v)
                    rem_edge!(graph, v, v)
                end
            end
        end
        # 3. Remove unneeded vertices.
        needed_vertices = find_necessary_vertices(graph, first_vertex,
                                                  last_vertex)
        first_vertex == 0 || @assert first_vertex in needed_vertices
        last_vertex == 0 || @assert last_vertex in needed_vertices
        if length(needed_vertices) < nv(graph)
            graph, vertex_mapping = induced_subgraph(graph, needed_vertices)
            reverse_vertex_mapping = Dict(vertex_mapping[v] => v
                                          for v in eachindex(vertex_mapping))
            if !issubset(initial_path, needed_vertices)
                # Initial path was invalid. Skip it.
                initial_path = []
            else
                # Renumber vertices in initial_path.
                initial_path = [reverse_vertex_mapping[v] for v in initial_path]
            end
            mapped_weights = map_weight_vertices(weights, reverse_vertex_mapping)
            if first_vertex != 0
                first_vertex = reverse_vertex_mapping[first_vertex]
            end
            if last_vertex != 0
                last_vertex = reverse_vertex_mapping[last_vertex]
            end
        end
    else
        if !is_directed(graph)
            error("Only directed graphs are supported for now. Convert your undirected graph to a directed representation or enable the preprocess option.")
        end
        if has_self_loops(graph)
            error("Self loops are not supported. Remove your self loops or enable the preprocess option.")
        end
    end

    if reduce_unbranched
        if graph === input_graph
            graph = copy(graph)
        end
        if mapped_weights === weights && are_weighted(mapped_weights)
            mapped_weights = deepcopy(mapped_weights)
        end
        graph, mapped_weights, vertex_mapping2, edge_mapping,
        first_vertex, last_vertex =
            reduce_unbranched_vertices!(graph, explicit_weights(mapped_weights,
                                                                graph),
                                        first_vertex, last_vertex)

        edge_mapping = map_edge_mapping(edge_mapping, vertex_mapping)
        vertex_mapping = vertex_mapping[vertex_mapping2]
    else
        edge_mapping = Dict{Tuple{Int, Int}, Vector{Int}}()
    end

    result = _find_longest_path(graph, mapped_weights,
                                first_vertex, last_vertex, vertex_mapping;
                                initial_path = initial_path, kwargs...)

    # Map back vertices of the best found path. Note: no mapping is
    # done of the internals of the result.
    result.longest_path = apply_mappings_to_path(result.longest_path,
                                                 vertex_mapping,
                                                 edge_mapping)
    result.weights = weights
    return result
end

function apply_mappings_to_path(path, vertex_mapping, edge_mapping)
    path = vertex_mapping[path]
    if isempty(edge_mapping)
        return path
    end
    for i = length(path):-1:2
        extra_vertices = get(edge_mapping, (path[i - 1], path[i]), Int[])
        if !isempty(extra_vertices)
            path = vcat(path[1:(i - 1)], extra_vertices, path[i:end])
        end
    end
    return path
end

# See comments for _pre_find_longest_path.
function find_necessary_vertices(graph, first_vertex, last_vertex)
    if first_vertex != last_vertex
        # Path
        vertices = find_reachable_vertices(v -> outneighbors(graph, v),
                                           first_vertex)
        if last_vertex != 0
            vertices2 = find_reachable_vertices(v -> inneighbors(graph, v),
                                                last_vertex)
            vertices = intersect(vertices, vertices2)
            # If there's no connection from first_vertex to
            # last_vertex we had an empty intersection. The
            # infeasibility will be redetected later, but at this
            # point we need to keep the first and last vertex.
            if isempty(vertices)
                vertices = [first_vertex, last_vertex]
            end
        end
    else
        # Cycle
        components = strongly_connected_components(graph)
        if first_vertex != 0
            # Can be shortened with `only` once support for Julia < 1.4
            # is dropped.
            matching_components = filter(c -> first_vertex in c, components)
            @assert length(matching_components) == 1
            vertices = first(matching_components)
        else
            if maximum(length.(components)) == 1
                vertices = first(components)
            else
                vertices = reduce(vcat, filter(c -> length(c) > 1, components))
            end
        end
    end
    return sort(vertices)
end

function find_reachable_vertices(f, first_vertex)
    queue = [first_vertex]
    vertices = empty(queue)
    while !isempty(queue)
        v = popfirst!(queue)
        if v ∉ vertices
            push!(vertices, v)
            append!(queue, f(v))
        end
    end
    return vertices
end

# TODO: With the exception of longest path ending anywhere, this kind
# of reduction can be done more aggressively to any vertex with a
# single ingoing or outgoing edge.
function reduce_unbranched_vertices!(graph, weights, first_vertex, last_vertex)
    edge_mapping = Dict{Tuple{Int, Int}, Vector{Int}}()
    keep_searching = true
    while keep_searching
        keep_searching = false
        for v in vertices(graph)
            v == first_vertex && continue
            v == last_vertex && continue
            i = inneighbors(graph, v)
            o = outneighbors(graph, v)
            isempty(i) && continue
            length(i) > 2 && continue
            isempty(o) && continue
            length(o) > 2 && continue
            length(union(i, o)) == 2 || continue
            for u in i
                w = first(o)
                if w == u
                    if length(o) == 1
                        continue
                    end
                    w = last(o)
                end
                if first_vertex > 0 && last_vertex == 0
                    if w == first_vertex || length(inneighbors(graph, w)) > 1
                        continue
                    end
                end

                # Reduce `u -> v -> w` to `u -> w`.

                # There might already be a an edge directly from u -> w
                # but that's fine. Just update the weight if the
                # reduced weight is higher.
                reduced_weight = weights.weights[(u, v)] + weights.weights[(v, w)]
                if haskey(weights.weights, (u, w))
                    weights.weights[(u, w)] = max(weights.weights[(u, w)],
                                                 reduced_weight)
                else
                    weights.weights[(u, w)] = reduced_weight
                end

                rem_edge!(graph, u, v)
                rem_edge!(graph, v, w)
                delete!(weights.weights, (u, v))
                delete!(weights.weights, (v, w))
                add_edge!(graph, u, w)
                edge_mapping[(u, w)] = vcat(get(edge_mapping, (u, v), Int[]),
                                            v,
                                            get(edge_mapping, (v, w), Int[]))
                delete!(edge_mapping, (u, v))
                delete!(edge_mapping, (v, w))
                keep_searching = true
            end
        end
    end

    remaining_vertices = filter(v -> (degree(graph, v) > 0 ||
                                      v == first_vertex ||
                                      v == last_vertex),
                                vertices(graph))
    graph, vertex_mapping = induced_subgraph(graph, remaining_vertices)
    forward_mapping = Dict(j => i for (i, j) in enumerate(vertex_mapping))
    weights = map_weight_vertices(weights, forward_mapping)
    if first_vertex != 0
        first_vertex = forward_mapping[first_vertex]
    end
    if last_vertex != 0
        last_vertex = forward_mapping[last_vertex]
    end

    return graph, weights, vertex_mapping, edge_mapping, first_vertex, last_vertex
end

map_weight_vertices(w::UnweightedPath, mapping) = w
map_weight_vertices(w::UnweightedCycle, mapping) = w

function map_weight_vertices(w::WeightedPath, mapping)
    w2 = deepcopy(w)
    map_weight_vertices!(w2.weights, w.weights, mapping)
    return w2
end

function map_weight_vertices(w::WeightedCycle, mapping)
    w2 = deepcopy(w)
    map_weight_vertices!(w2.weights, w.weights, mapping)
    return w2
end

function map_weight_vertices!(new_weights::Dict, old_weights::Dict, mapping)
    empty!(new_weights)
    for (k, v) in old_weights
        new_weights[(mapping[k[1]], mapping[k[2]])] = v
    end
end

function map_edge_mapping(edge_mapping, vertex_mapping)
    out_edge_mapping = empty(edge_mapping)
    for (k, v) in edge_mapping
        out_edge_mapping[(vertex_mapping[k[1]],
                          vertex_mapping[k[2]])] = vertex_mapping[v]
    end
    return out_edge_mapping
end

# The main search function for both paths and cycles. At this point
# cycle search is indicated by `first_vertex == last_vertex` and by
# the type of `weights`.
function _find_longest_path(graph, weights::AbstractWeightedPath,
                            first_vertex, last_vertex, vertex_mapping;
                            initial_path = Int[],
                            lower_bound = -Inf,
                            upper_bound = Inf,
                            solver_mode = "ip",
                            cycle_constraint_mode = "cutset",
                            initial_cycle_constraints = 0,
                            max_iterations = typemax(Int),
                            time_limit = typemax(Int),
                            solver_time_limit = 10,
                            max_gap = 0,
                            use_ip_warmstart = true,
                            log_level = 1,
                            new_longest_path_callback = x -> nothing,
                            iteration_callback = print_iteration_data)
    @assert xor(first_vertex == last_vertex, !are_cycle_weights(weights))
    # TODO: Convert these to proper validation of user input.
    @assert(solver_mode ∈ ["lp", "lp+ip", "ip"],
            "solver_mode must be one of \"lp\", \"lp+ip\", \"ip\"")
    @assert(cycle_constraint_mode ∈ ["cycle", "cutset", "both"], 
            "cycle_constraint_mode must be one of \"cycle\", \"cutset\", \"both\"")

    if !(0 <= first_vertex <= nv(graph))
        error("The first vertex is outside the vertex range of the graph.")
    end

    if !(0 <= last_vertex <= nv(graph))
        error("The last vertex is outside the vertex range of the graph.")
    end

    check_if_weights_are_complete(graph, weights)

    # Possibly improve the provided upper bound with a trivial upper
    # bound computed from the weights of the problem.
    upper_bound = min(upper_bound, trivial_upper_bound(graph, weights))

    # Find a first solution and establish a lower bound from that.  If
    # the user provided initial path is valid, use that. Otherwise
    # find an arbitrary solution. If none can be found, abort the
    # search and report that the problem is infeasible.
    #
    # This needs to be done differently for the different kinds of
    # searches, so split it up in the different cases.

    if first_vertex != last_vertex
        # Path search.
        if (isempty(initial_path)
            || !is_simple_path(graph, initial_path)
            || initial_path[1] != first_vertex
            || (initial_path[end] != last_vertex
                && last_vertex != 0))
            # The provided initial path is not valid. Look for an
            # arbitrary valid solution.
            if last_vertex == 0
                path = [first_vertex]
            else
                path = get_path(graph, first_vertex, last_vertex)
            end
            if isempty(path)
                # Infeasible problem.
                path = Int[]
                lb = ub = path_length(path, weights)
                return LongestPathOrCycle(weights, lb, ub, path,
                                          Dict{String, Any}())
            else
                new_longest_path_callback(vertex_mapping[path])
                initial_path = path
            end
        end
    else
        # Cycle search.
        if (isempty(initial_path)
            || !is_simple_cycle(graph, initial_path)
            || (first_vertex ∉ initial_path && first_vertex != 0))
            # The provided initial cycle is not valid. Look for an
            # arbitrary valid solution.
            if first_vertex == 0
                cycle = get_cycle(graph)
            else
                cycle = get_cycle(graph, first_vertex)
            end

            if isempty(cycle)
                # Infeasible problem.
                path = Int[]
                lb = ub = path_length(path, weights)
                return LongestPathOrCycle(weights, lb, ub, path,
                                          Dict{String, Any}())
            else
                new_longest_path_callback(vertex_mapping[cycle])
                initial_path = cycle
            end
        end
    end

    # The initial path can now be trusted. Use it to possibly improve
    # the lower bound.
    best_path = initial_path
    lower_bound = max(lower_bound, path_length(best_path, weights))

    O, edges = OptProblem(graph, weights, first_vertex, last_vertex)
    reverse_edges = Dict(edges[k] => k for k = 1:length(edges))

    if initial_cycle_constraints > 1
        cycles = simplecycles_limited_length(graph, initial_cycle_constraints)
        if first_vertex == last_vertex == 0
            best_path = filter_out_longest_cycle!(best_path, weights, cycles,
                                                  new_longest_path_callback,
                                                  vertex_mapping)
        end
        constrain_cycles!(O, weights, cycles, edges, cycle_constraint_mode,
                          first_vertex, best_path)
    end

    # Due to a Cbc bug, ip warmstart cannot be used with negative weights.
    # TODO: Report bug upstream and link the issue here.
    if use_ip_warmstart && are_weighted(weights) && any(values(weights.weights) .< 0)
        use_ip_warmstart = false
        if log_level >= 1
            println("Turning off use_ip_warmstart to work around a Cbc bug.")
        end
    end

    solution = nothing
    
    start_time = time()
    main_path = Int[]
    cycles = Vector{Int}[]
    
    for iteration = 1:max_iterations
        if are_integer_weights(weights)
            max_gap = min(max_gap, upper_bound - lower_bound - 1)
        else
            max_gap = min(max_gap, 0.99 * (upper_bound - lower_bound))
        end

        solver_time = min(solver_time_limit, time_limit - (time() - start_time))
        if solver_time < solver_time_limit / 2
            break
        end

        if solver_mode == "lp" || (solver_mode == "lp+ip" && iteration % 2 == 1)
            solution = solve_LP(O)
            objbound = solution.objval
        else
            path_edges = path_to_edge_variables(best_path, reverse_edges,
                                                first_vertex == last_vertex)
            solution = solve_IP(O, path_edges, use_ip_warmstart,
                                seconds = solver_time,
                                allowableGap = max_gap,
                                logLevel = max(0, log_level - 1))
            objbound = solution.attrs[:objbound]
        end

        if are_integer_weights(weights)
            upper_bound = min(upper_bound,
                              floor(Int, round(objbound, digits = 3)))
        else
            upper_bound = min(upper_bound, objbound)
        end

        main_path, cycles = extract_paths(graph, edges,
                                          reverse_edges,
                                          first_vertex, last_vertex,
                                          solution.sol)

        if first_vertex == last_vertex
            if first_vertex == 0
                push!(cycles, main_path)
                best_path = filter_out_longest_cycle!(best_path, weights,
                                                      cycles,
                                                      new_longest_path_callback,
                                                      vertex_mapping)
            end
            lower_bound = max(lower_bound, path_length(best_path, weights))
        end

        lower_bound = max(lower_bound, path_length(main_path, weights))
        if path_length(main_path, weights) > path_length(best_path, weights)
            best_path = main_path
            new_longest_path_callback(vertex_mapping[best_path])
        end

        iteration_data = (log_level = log_level,
                          iteration = iteration,
                          elapsed_time = time() - start_time,
                          lower_bound = lower_bound,
                          upper_bound = upper_bound,
                          max_gap = max_gap,
                          solver_time = solver_time,
                          num_constraints = length(O.cycle_constraints),
                          solution = solution,
                          objbound = objbound,
                          best_path = best_path,
                          main_path = main_path,
                          cycles = cycles,
                          vertex_mapping = vertex_mapping)

        if !iteration_callback(iteration_data)
            break
        end

        if lower_bound >= upper_bound || iteration == max_iterations
            break
        end

        if solution.attrs[:solver] == :lp
            cutsets = find_fractional_cutsets(graph, edges, reverse_edges,
                                              solution.sol)
            # When searching for longest cycle anywhere, we have to be
            # sure that no cutset will eliminate the optimal solution.
            # This is safe but quite conservative.
            #
            # TODO: Relax this test to allow more valid cutsets.
            #
            # Note: We don't need to do this when searching for a
            # cycle through a specified vertex since
            # `constrain_cycles!` will filter out cycles through that
            # vertex.
            if first_vertex == last_vertex == 0
                cutsets = filter(x -> path_length(x, weights) < path_length(best_path, weights), cutsets)
            end

            append!(cycles, cutsets)
        end

        # If no cycles or cutsets have been found there are no new
        # constraints to add. In LP mode there's no point continuing
        # since we can expect to find ourselves in exactly the same
        # situation in the next iteration too. In IP mode this can
        # only happen due to time limit restrictions (since we didn't
        # break out earlier with an optimal solution) so we increase
        # the solver time limit and try again. In LP+IP mode we do
        # nothing on an LP iteration and increase solver time limit on
        # an IP iteration.
        if length(cycles) == 0
            if solver_mode == "lp"
                break
            elseif solution.attrs[:solver] == :ip
                solver_time_limit = ceil(1.6 * solver_time_limit)
            end
            continue
        end

        selected_cycles = select_cycles(cycles)
        constrain_cycles!(O, weights, selected_cycles, edges,
                          cycle_constraint_mode, first_vertex, best_path)
    end

    return LongestPathOrCycle(weights,
                              promote(lower_bound, upper_bound)...,
                              best_path,
                              Dict("O" => O, "edges" => edges,
                                   "last_path" => main_path,
                                   "last_cycles" => cycles,
                                   "last_solution" => solution,
                                   "vertex_mapping" => vertex_mapping))
end

function check_if_weights_are_complete(graph, weights)
    if are_weighted(weights)
        for edge in edges(graph)
            if !haskey(weights.weights, Tuple(edge))
                error("Weights are provided but not specified for all edges.")
            end
        end
    end
end

path_length(path, weights::UnweightedPath) = length(path) - !isempty(path)
path_length(path, weights::UnweightedCycle) = length(path)

function path_length(path, weights::Dict{Tuple{Int, Int}, <:Any})
    L = 0
    for k = 2:length(path)
        L += weights[(path[k - 1], path[k])]
    end
    return L
end

function path_length(path, weights::WeightedPath)
    # An empty path vector means no solution and length is reported as 0.
    isempty(path) && return 0
    return path_length(path, weights.weights)
end

function path_length(cycle, weights::WeightedCycle)
    # An empty cycle vector means no solution and length is reported as 0.
    isempty(cycle) && return 0
    # There exist no cycles of length 1.
    @assert length(cycle) != 1
    # Weight of going back from the end to the beginning.
    w = weights.weights[(cycle[end], cycle[1])]
    return path_length(cycle, weights.weights) + w
end

"""
    is_path(graph, path)

Determine whether the vector of vertices `path` forms a path through
`graph`.
"""
function is_path(graph, path)
    e = edges(g)
    return all((path[i-1], path[i]) ∈ e for i = 2:length(path))
end

"""
    is_cycle(graph, path)

Determine whether the vector of vertices `path` forms a cycle through
`graph`, where the cycle is closed by an edge from the last vertex in
`path` to the first vertex in `path`.
"""
function is_cycle(graph, path)
    return is_path(graph, path) && (path[end], path[1]) ∈ graph
end

"""
    is_simple_path(graph, path)

Determine whether the vector of vertices `path` forms a simple path
through `graph`.
"""
function is_simple_path(graph, path)
    length(unique(path)) == length(path) || return false
    return is_path(graph, path)
end

"""
    is_simple_cycle(graph, path)

Determine whether the vector of vertices `path` forms a simple cycle
through `graph`, where the cycle is closed by an edge from the last
vertex in `path` to the first vertex in `path`.
"""
function is_simple_cycle(graph, path)
    return is_simple_path(graph, path) && (path[end], path[1]) ∈ graph
end

function print_iteration_data(data)
    if data.log_level > 0
        @printf("%3d %5d ", data.iteration, round(Int, data.elapsed_time))
        print("[$(data.lower_bound) $(data.upper_bound)] $(data.max_gap) ")
        println("$(data.num_constraints) $(data.solution.status) $(data.solution.objval) $(data.objbound)")
    end
    return true
end

function trivial_upper_bound(graph, weights)
    if !are_weighted(weights)
        return nv(graph)
    else
        w = sort(collect(values(weights.weights)), lt = >)
        return sum(max.(0, w[1:min(end, nv(graph))]))
    end
end

# Convert a path to the corresponding setting of binary edge variables.
function path_to_edge_variables(path, reverse_edges, is_cycle)
    e = zeros(Float64, length(reverse_edges))
    for k = 1:length(path) - 1
        v1, v2 = path[k], path[k + 1]
        e[reverse_edges[(v1, v2)]] = 1
    end
    if is_cycle
        e[reverse_edges[(path[end], path[1])]] = 1
    end
    return e
end

struct CycleConstraint
    A::SparseVector
    lb::Int
    ub::Int
end

# lb and ub are lower and upper bounds on the constraints in A.
# l and u are lower and upper bounds on the variables, i.e. the edges.
mutable struct OptProblem
    A::SparseMatrixCSC{Int64,Int64}
    c::Vector{Float64}
    lb::Vector{Int}
    ub::Vector{Int}
    l::Vector{Int}
    u::Vector{Int}
    vartypes::Vector{Symbol}
    cycle_constraints::Vector{CycleConstraint}
end

function OptProblem(graph, weights,
                    first_vertex = 1,
                    last_vertex = 0)
    # N is number of vertices in graph problem.
    # M is number of edges.
    N = nv(graph)
    M = ne(graph)
    # Weighted search for a cycle anywhere needs an extra
    # constraint. See the comment where `N₀` is used.
    N₀ = (first_vertex == last_vertex == 0) && are_weighted(weights)

    A = spzeros(Int, 2 * N + N₀, M)
    lb = zeros(Int, 2 * N + N₀)
    ub = zeros(Int, 2 * N + N₀)
    l = zeros(Int, M)
    u = ones(Int, M)
    vartypes = fill(:Bin, M)
    edge_variables = Tuple{Int, Int}[]
    # There are two constraints per vertex, numbered 2n-1 and 2n,
    # where n is the number of the vertex.
    #
    # The odd-numbered constraints sums the values of the incoming
    # edges to the vertex. The even-numbered constraints sums the
    # values of the outgoing edges and subtracts the values of the
    # incoming edges.
    #
    # The bounds on these constraints depend on the type of search as
    # specified in the tables below.
    #
    # 1. Search from a specified start point to a specified end point,
    #    i.e. 0 != first_vertex != last_vertex != 0.
    #
    #       first vertex  last vertex  other vertices
    # odd   0              1           [0, 1]
    # even  1             -1            0
    #
    # 2. Search for a cycle through a specified vertex,
    #    i.e. first_vertex == last_vertex != 0.
    #
    #       first vertex  last vertex  other vertices
    # odd   1             N/A          [0, 1]
    # even  0             N/A           0
    #
    # 3. Search from a specified start point to an arbitrary end point,
    #    i.e. first_vertex != 0 and last_vertex == 0.
    #
    #       first vertex  last vertex  other vertices
    # odd    0            N/A          [ 0, 1]
    # even  [0, 1]        N/A          [-1, 0]
    #
    # 4. Search for a cycle anywhere,
    #    i.e. first_vertex == last_vertex == 0.
    #
    #       first vertex  last vertex  other vertices
    # odd   N/A           N/A          [0, 1]
    # even  N/A           N/A           0

    m = 0
    for n = 1:N
        # Set up the coefficients of the constraints. This relies on
        # the ordering of edge variables matching the iteration over
        # vertices and outneighbors.
        for k in outneighbors(graph, n)
            push!(edge_variables, (n, k))
            m += 1
            A[2 * k - 1, m] = 1  # Incoming edge to vertex k.
            A[2 * n, m] = 1      # Outgoing edge from vertex n.
            A[2 * k, m] = -1     # Incoming edge to vertex k.
        end
        # Set Constraint bounds for "other vertices". Bounds for the
        # first and last vertices will be corrected after the loop.
        # Note: bounds are already initialized to 0, so we could skip
        # filling in zero values again but this is not time critical,
        # so let's be clear and explicit.
        lb[2 * n - 1] = 0
        ub[2 * n - 1] = 1
        if first_vertex != 0 && last_vertex == 0
            lb[2 * n] = -1
        else
            lb[2 * n] = 0
        end
        ub[2 * n] = 0
    end
    @assert m == M

    # Fill in bounds for first and last vertices, if specified.
    if first_vertex != 0
        if last_vertex == first_vertex
            lb[2 * first_vertex - 1] = 1
        else
            lb[2 * first_vertex - 1] = 0
            ub[2 * first_vertex - 1] = 0
            ub[2 * first_vertex] = 1
            if last_vertex != 0
                lb[2 * first_vertex] = 1
                lb[2 * last_vertex - 1] = 1
                ub[2 * last_vertex - 1] = 1
                lb[2 * last_vertex] = -1
                ub[2 * last_vertex] = -1
            else
                lb[2 * first_vertex] = 0
            end
        end
    end

    # When searching for a cycle anywhere, the constraints so far
    # allow the trivial all-zero solution, corresponding to an invalid
    # no-vertex cycle. This is usually not a problem since it would be
    # a very short cycle, but if the problem is weighted with negative
    # weights, it would yield a false solution. To overcome this, add
    # the constraint that the sum of all edges is at least one. This
    # is only needed when the optimal solution is negative but always
    # valid, so add it for weighted problems regardless of weights.
    if N₀
        A[2 * N + N₀, :] .= 1
        lb[2 * N + N₀] = 1
        ub[2 * N + N₀] = M
    end

    if !are_weighted(weights)
        c = ones(Float64, M)
    else
        c = Float64[weights.weights[e] for e in edge_variables]
    end

    return (OptProblem(A, c, lb, ub, l, u, vartypes, CycleConstraint[]),
            edge_variables)
end

function solve_LP(O::OptProblem; kw...)
    model = LinearQuadraticModel(ClpSolver(;kw...))
    A, lb, ub = add_cycle_constraints_to_formulation(O)
    loadproblem!(model, A, O.l, O.u, O.c, lb, ub, :Max)
    optimize!(model)
    attrs = Dict()
    attrs[:redcost] = getreducedcosts(model)
    attrs[:lambda] = getconstrduals(model)
    attrs[:solver] = :lp
    solution = MathProgBase.HighLevelInterface.LinprogSolution(status(model), getobjval(model), getsolution(model), attrs)
end

function solve_IP(O::OptProblem, initial_solution = Int[],
                  use_warmstart = true; kw...)
    model = LinearQuadraticModel(CbcSolver(;kw...))
    A, lb, ub = add_cycle_constraints_to_formulation(O)
    loadproblem!(model, A, O.l, O.u, O.c, lb, ub, :Max)
    setvartype!(model, O.vartypes)
    original_stdout = stdout
    if !isempty(initial_solution) && use_warmstart
        setwarmstart!(model, initial_solution)
    end
    optimize!(model)
   
    attrs = Dict()
    attrs[:objbound] = getobjbound(model)
    attrs[:solver] = :ip
    solution = MathProgBase.HighLevelInterface.MixintprogSolution(status(model), getobjval(model), getsolution(model), attrs)
    return solution
end

function add_cycle_constraints_to_formulation(O::OptProblem)
    n = length(O.cycle_constraints)
    A = spzeros(Int, size(O.A, 1) + n, size(O.A, 2))
    lb = zeros(Int, length(O.lb) + n)
    ub = zeros(Int, length(O.ub) + n)
    A[1:size(O.A, 1), :] = O.A
    lb[1:length(O.lb)] = O.lb
    ub[1:length(O.ub)] = O.ub
    i = size(O.A, 1) + 1
    for c in O.cycle_constraints
        A[i,:] = c.A
        lb[i] = c.lb
        ub[i] = c.ub
        i += 1
    end
    return A, lb, ub
end

# Find a path or cycle starting at `v`.
#
# This function relies on the assumption that each vertex has at most
# one incoming and at most one outgoing edge.
function follow_path(graph, v)
    path = [v]
    while true
        n = outneighbors(graph, path[end])
        if isempty(n)
            break
        elseif n[1] == v
            return path, true
        end
        push!(path, n[1])
    end

    return path, false
end

# Find a path from `first_vertex`, a cycle through `first_vertex`, or
# a long cycle. Complement with additional cycles.
#
# By filtering out edges with value less than 0.51 it is guaranteed
# that each vertex has at most one incoming and at most one outgoing
# edge. As a corollary every strongly connected component with more
# than one vertex contains exactly one cycle.
function extract_paths(graph, edges, reverse_edges, first_vertex, last_vertex,
                       solution)
    graph2 = SimpleDiGraph(nv(graph))
    for i = 1:length(solution)
        if solution[i] >= 0.51
            add_edge!(graph2, edges[i]...)
        end
    end

    cutsets = filter(x -> length(x) > 1, strongly_connected_components(graph2))

    if first_vertex != last_vertex
        main_path, is_cycle = follow_path(graph2, first_vertex)
        @assert !is_cycle
        if 0 != last_vertex != main_path[end]
            main_path = Int[]
        end
    else
        if isempty(cutsets)
            return Int[], cutsets
        end

        if first_vertex != 0
            cutsets = filter(x -> first_vertex ∉ x, cutsets)
            main_path, is_cycle = follow_path(graph2, first_vertex)
            if !is_cycle
                main_path = Int[]
            end
        else
            _, i = findmax(length.(cutsets))
            main_path, is_cycle = follow_path(graph2, cutsets[i][1])
            @assert is_cycle
            if i < length(cutsets)
                cutsets[i] = pop!(cutsets)
            else
                pop!(cutsets)
            end
        end
    end

    return main_path, cutsets
end


function find_fractional_cutsets(graph, edges, reverse_edges, solution)
    graph2 = SimpleDiGraph(nv(graph))
    for i = 1:length(solution)
        if 0.01 < solution[i] < 0.99
            add_edge!(graph2, edges[i]...)
        end
    end

    cutsets = Vector{Int}[]
    for component in strongly_connected_components(graph2)
        if cycle_constraints_margin(graph, reverse_edges,
                                    solution, component) < -0.1
            push!(cutsets, component)
        end
    end

    return cutsets
end

function cycle_constraints_margin(graph, reverse_edges, solution, vertices)
    total_internal_flow = 0.0
    total_external_inflow = 0.0
    max_internal_inflow = 0.0
    for v in vertices
        internal_inflow = 0.0
        for n in inneighbors(graph, v)
            e = reverse_edges[(n, v)]
            if solution[e] > 0
                if n ∈ vertices
                    total_internal_flow += solution[e]
                    internal_inflow += solution[e]
                else
                    total_external_inflow += solution[e]
                end
            end
        end
        if internal_inflow > max_internal_inflow
            max_internal_inflow = internal_inflow
        end
    end

    margin = min(length(vertices) - 1 - total_internal_flow,
                 total_external_inflow - max_internal_inflow)

    return margin
end

# When searching for longest cycle, a new best cycle may have appeared
# among the detected cycles that we are otherwise going to eliminate
# with additional constraints. In that case, update the best cycle and
# remove it from cycles to be eliminated.
#
# It could also happen that the current best cycle is among the cycles
# to be eliminated. We could try to detect whether this is the case
# and filter it out, but a simpler solution is to just switch to a new
# cycle of the same length as the previous best and filter that one
# out, which is trivial.
function filter_out_longest_cycle!(best_path, weights,
                                   cycles, new_longest_path_callback,
                                   vertex_mapping)
    if isempty(cycles)
        return best_path
    end

    cycle_lengths = path_length.(cycles, Ref(weights))
    longest_cycle = maximum(cycle_lengths)

    # Current best cycle is longer than all cycles to be eliminated.
    # Nothing needs to be done.
    if longest_cycle < path_length(best_path, weights)
        return best_path
    end

    # Find one of the longest cycles in cycles.
    i = findlast(cycle_lengths .== longest_cycle)
    cycle = cycles[i]

    # The former best path is now just a cycle among the others and a
    # candidate for a constraint.
    cycles[i] = best_path

    # Replace the current best cycle with the new cycle (by returning it).
    if path_length(cycle, weights) > path_length(best_path, weights)
        new_longest_path_callback(vertex_mapping[best_path])
    end
    return cycle
end


function select_cycles(cycles)
    cutoff_length = max(minimum(length.(cycles)), 12)
    return filter(c -> length(c) <= cutoff_length, cycles)
end

# Add constraints derived from cycles in the graph. (Technically it
# doesn't have to be cycles, any arbitrary set of vertices not
# including the first vertex is fine.)
#
# There are two kinds of constraints, internal and external.
#
# The primary internal constraint is to disrupt cycles by saying that
# the sum of the edge variables in the cycle must be smaller than the
# length of the cycle. However, we can strengthen this constraint in
# two ways. First, instead of "less than the length" we use "less than
# or equal to length - 1". For integer solutions it's the same thing
# but for fractional solutions it's stronger, and thus improves the
# bounds of the LP relaxation. Second, instead of only summing edge
# variables along the cycle we can sum all edges between vertices in
# the cycle. This is also strictly stronger and helps to tighten the
# LP bound.
#
# The external constraints are generalized cutset inequalities in the
# terminology of Taccari. For each vertex in the cycle, the ingoing
# edges to the vertex should sum to less than or equal to the sum of
# the ingoing edges to the entire cycle.
#
# Return the number of added constraints.
function constrain_cycles!(O::OptProblem, weights, cycles, edges,
                           cycle_constraint_mode, first_vertex, best_path)
    previous_number_of_constraints = length(O.cycle_constraints)
    for cycle in cycles
        # We need to filter out some cutsets to make sure we don't
        # lose any best solution or end up with an infeasible system.
        if first_vertex != 0
            # If searching for a path or for a cycle through a
            # specified vertex, the first_vertex must not be part of
            # any cutset constraint.
            #
            # TODO: Actually, it's ok to filter out shorter cycles
            # through the first vertex when searching for cycles. It's
            # not strictly necessary in order to find the optimal
            # solution but it may improve the efficiency.
            if first_vertex ∈ cycle
                continue
            end
        else
            # When searching for a cycle anywhere, we don't want to
            # remove the current best solution but we also can't
            # remove longer cutsets in case one of those happens to
            # contain the longest cycle.
            if path_length(cycle, weights) == path_length(best_path, weights)
                if isempty(setdiff(cycle, best_path))
                    continue
                end
            elseif path_length(cycle, weights) > path_length(best_path, weights)
                continue
            end
        end

        if cycle_constraint_mode == "cycle" || cycle_constraint_mode == "both"
            A = spzeros(Int, size(O.A, 2))
            for i = 1:length(edges)
                v1, v2 = edges[i]
                if v1 ∈ cycle && v2 ∈ cycle
                    A[i] = 1
                end
            end
            lb = 0
            ub = length(cycle) - 1
            push!(O.cycle_constraints, CycleConstraint(A, lb, ub))
        end

        if cycle_constraint_mode == "cutset" || cycle_constraint_mode == "both"
            # Find all incoming edges to the cycle.
            # TODO: Check whether this can be computed more efficiently.
            incoming_edges_to_cycle = findall([edge[1] ∉ cycle && edge[2] ∈ cycle for edge in edges])
            # Add one constraint for each vertex in the cycle.
            for vertex in cycle
                A = spzeros(Int, size(O.A, 2))
                # TODO: Check whether this can be computed more efficiently.
                incoming_edges_to_vertex = findall([edge[2] == vertex for edge in edges])
                # The number of incoming edges to the vertex must be
                # smaller or equal to the number of incoming edges to
                # the cycle. The point of this is that without
                # incoming edges to the cycle, it can't have any
                # internal edges.
                A[incoming_edges_to_cycle] .= 1
                A[incoming_edges_to_vertex] .-= 1
                lb = 0
                ub = length(cycle)
                push!(O.cycle_constraints, CycleConstraint(A, lb, ub))
            end
        end
    end

    return length(O.cycle_constraints) - previous_number_of_constraints
end

# Borrowed from https://github.com/JuliaGraphs/LightGraphs.jl/pull/1095
# until is has been merged and released.
function get_cycle(g::AbstractGraph)
    return get_path_or_cycle(g, vertices(g), 0, true, false)
end

function get_cycle(g::AbstractGraph, v)
    return get_path_or_cycle(g, v, v, true, false)
end

function get_path(g::AbstractGraph, v::Integer, w::Integer)
    return get_path_or_cycle(g, v, w, false, false)
end

function get_path_or_cycle(g::AbstractGraph{T}, sources, target, find_cycle,
                           only_detect_cycle) where T
    vcolor = zeros(UInt8, nv(g))
    path = Vector{T}()
    for source in sources
        vcolor[source] != 0 && continue
        push!(path, source)
        vcolor[source] = 1
        while !isempty(path)
            u = path[end]
            w = T(0)
            for n in outneighbors(g, u)
                if vcolor[n] == 0
                    w = n
                    break
                elseif vcolor[n] == 1 && find_cycle && (target == 0 || target == n)
                    if !only_detect_cycle
                        while path[1] != n
                            popfirst!(path)
                        end
                    end
                    return path
                end
            end
            if w != 0
                push!(path, w)
                if w == target
                    return path
                end
                vcolor[w] = 1
            else
                vcolor[u] = 2
                pop!(path)
            end
        end
    end

    return path
end
