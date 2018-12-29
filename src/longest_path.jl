export LongestPath, longest_path

using MathProgBase
using MathProgBase.SolverInterface
using Clp
using Cbc
using LightGraphs
using SparseArrays
using Printf
using Random
using Suppressor

"""
    LongestPath

Type used for the return values of `longest_path`. See the function
documentation for more information.
"""
mutable struct LongestPath
    lower_bound::Int
    upper_bound::Int
    longest_path::Vector{Int}
    internals::Dict{String, Any}
end

function Base.show(io::IO, x::LongestPath)
    println(io, "Longest path with bounds [$(x.lower_bound), $(x.upper_bound)] and a recorded path of length $(max(0, length(x.longest_path) - 1)).")
end

"""
    longest_path(graph)

Find the longest simple path in `graph` starting from vertex number
one and ending anywhere. No vertex may be visited more than once.
Given sufficient time and memory, this will succeed in finding the
longest path, but since finding the longest path is an NP-hard
problem, the required time will grow quickly with the graph size.

For the time being, `graph` must be a **directed** graph from the
`LightGraphs` package. The algorithm works for undirected graphs as
well, but you must first represent it as a directed graph.

    longest_path(graph; kwargs)

By adding keyword arguments it is possible to obtain non-optimal
solutions and bounds in shorter time than the full solution. It is
also possible to modify the problem specification.

* `first_vertex`: Start vertex for the path. Default is 1. **Not yet
  implemented**.

* `last_vertex`: End vertex for the path. Default is 0, meaning
  anywhere. **Not yet implemented**.

* `initial_path`: Search can be warmstarted by providing a valid path
  as a vector of vertices. Default is an empty vector.

* `lower_bound`: User provided lower bound. Search will stop when the
  upper bound reaches `lower_bound`, even if no path of that length
  has been found. Default is the number of edges in `initial_path`.
  The provided `lower_bound` will be ignored if a stronger bound is
  found during search.

* `upper_bound`: User provided upper bound. Search will stop when the
  lower bound reaches `upper_bound`, even if longer paths exist.
  Default is the number of vertices in the graph. The provided
  `upper_bound` will be ignored if a stronger bound is found during
  search.

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

The return value is of the `LongestPath` type and contains the
following fields:

* `lower_bound`: Lower bound for the length of the longest path.

* `upper_bound`: Upper bound for the length of the longest path.

* `longest_path`: Vector of the vertices in the longest found path.

* `internals`: Dict containing a variety of information about the search.

Note: path lengths are reported by number of edges, not number of vertices.
"""
function longest_path(graph;
                      first_vertex = 1,
                      last_vertex = 0,
                      initial_path = Int[],
                      lower_bound = length(initial_path) - 1,
                      upper_bound = nv(graph),
                      solver_mode = "ip",
                      cycle_constraint_mode = "cutset",
                      initial_cycle_constraints = 0,
                      max_iterations = typemax(Int),
                      time_limit = typemax(Int),
                      solver_time_limit = 10,
                      max_gap = 0,
                      log_level = 1,
                      new_longest_path_callback = x -> nothing,
                      iteration_callback = print_iteration_data)
    @assert(solver_mode ∈ ["lp", "lp+ip", "ip"],
            "solver_mode must be one of \"lp\", \"lp+ip\", \"ip\"")
    @assert(cycle_constraint_mode ∈ ["cycle", "cutset", "both"], 
            "cycle_constraint_mode must be one of \"cycle\", \"cutset\", \"both\"")

    if !is_directed(graph)
        error("Only directed graphs are supported for now. Convert your undirected graph to a directed representation.")
    end

    if first_vertex != 1 || last_vertex != 0
        error("At the moment paths have to start at vertex 1 and be open-ended.")
    end

    # TODO: Check why this really is necessary.
    if isempty(outneighbors(graph, first_vertex))
        return LongestPath(0, 0, [1], Dict())
    end
                
    # Terribly ugly but these lines forces a trace message from
    # `setwarmstart!`, after which all following ones can be suppressed.
    # Without these three lines, output from `setwarmstart!` is only
    # momentarily suppressed but comes back slightly delayed.
    # See https://github.com/JuliaOpt/Cbc.jl/issues/78.
    if log_level < 2 && solver_mode != "lp"
        println("Please ignore this output. At the moment it's necessary in order to suppress later output.")
        println("---------------------")
        model = LinearQuadraticModel(CbcSolver(logLevel=0))
        loadproblem!(model, ones(2,2), zeros(2), ones(2), ones(2), zeros(2), ones(2), :Max)
        setwarmstart!(model, zeros(2))
        println("---------------------")
    end

    O = OptProblem(graph, first_vertex, last_vertex)
    edges = get_all_edges(graph)
    reverse_edges = Dict(edges[k] => k for k = 1:length(edges))

    if initial_cycle_constraints > 1
        cycles = simplecycles_limited_length(graph, initial_cycle_constraints)
        constrain_cycles!(O, cycles, edges, nothing, cycle_constraint_mode)
    end

    solution = nothing
    
    start_time = time()
    best_path = initial_path
    main_path = Int[]
    cycles = Vector{Int}[]
    
    for iteration = 1:max_iterations
        max_gap = min(max_gap, upper_bound - lower_bound - 1)

        solver_time = min(solver_time_limit, time_limit - (time() - start_time))
        if solver_time < solver_time_limit / 2
            break
        end

        path_edges = path_to_edge_variables(best_path, reverse_edges)

        if solver_mode == "lp" || (solver_mode == "lp+ip" && iteration % 2 == 1)
            solution = solve_LP(O)
            objbound = solution.objval
        else
            solution = solve_IP(O, path_edges, seconds = solver_time,
                                 allowableGap = max_gap,
                                 logLevel = max(0, log_level - 1))
            objbound = solution.attrs[:objbound]
        end

        upper_bound = min(upper_bound, floor(Int, round(objbound, digits = 3)))

        main_path, cycles, fragments = extract_paths(graph, edges,
                                                     reverse_edges,
                                                     first_vertex, solution.sol)

        lower_bound = max(lower_bound, length(main_path) - 1)
        if length(main_path) > length(best_path)
            best_path = main_path
            new_longest_path_callback(best_path)
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
                          cycles = cycles)
        if !iteration_callback(iteration_data)
            break
        end
        
        if lower_bound >= upper_bound || iteration == max_iterations
            break
        end

        if solution.attrs[:solver] == :lp
            append!(cycles, find_fractional_cutsets(graph, edges,
                                                    reverse_edges, solution.sol,
                                                    fragments, first_vertex))
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
        constrain_cycles!(O, selected_cycles, edges, solution.sol,
                          cycle_constraint_mode)
    end

    return LongestPath(lower_bound, upper_bound,
                       best_path,
                       Dict("O" => O, "edges" => edges,
                            "last_path" => main_path, "last_cycles" => cycles,
                            "last_solution" => solution))
end

function print_iteration_data(data)
    if data.log_level > 0
        @printf("%3d %5d ", data.iteration, round(Int, data.elapsed_time))
        print("[$(data.lower_bound) $(data.upper_bound)] $(data.max_gap) ")
        println("$(data.num_constraints) $(data.solution.status) $(data.solution.objval) $(data.objbound)")
    end
    return true
end

# Convert a path to the corresponding setting of binary edge variables.
function path_to_edge_variables(path, reverse_edges)
    e = zeros(Float64, length(reverse_edges))
    for k = 1:length(path) - 1
        v1, v2 = path[k], path[k + 1]
        e[reverse_edges[(v1, v2)]] = 1
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
    c::Vector{Int}
    lb::Vector{Int}
    ub::Vector{Int}
    l::Vector{Int}
    u::Vector{Int}
    vartypes::Vector{Symbol}
    cycle_constraints::Vector{CycleConstraint}
end

function OptProblem(graph,
                    first_vertex = 1,
                    last_vertex = 0)
    # N is number of vertices in graph problem.
    # M is number of edges.
    N = nv(graph)
    M = ne(graph)
    c = ones(Int, M)
    A = spzeros(Int, 2 * N, M)
    lb = zeros(Int, 2 * N)
    ub = zeros(Int, 2 * N)
    l = zeros(Int, M)
    u = ones(Int, M)
    vartypes = fill(:Bin, M)
    # TODO:
    # * Support different options for start and end vertices.
    #
    # There are two constraints per vertex, numbered 2n-1 and 2n,
    # where n is the number of the vertex. The odd-numbered
    # constraints sums the values of the incoming edges to the vertex,
    # which must be between 0 and 1, except for the first vertex where
    # the sum must be 0. The even-numbered constraints sums the values
    # of the outgoing edges and subtracts the values of the incoming
    # edges. This must be between -1 and 0 for all vertices except the
    # first, which must be between 0 and 1.
    m = 0
    for n = 1:N
        lb[2 * n - 1] = 0
        ub[2 * n - 1] = Int(n > 1)
        lb[2 * n] = -1
        for k in outneighbors(graph, n)
            m += 1
            A[2 * n, m] = 1
            A[2 * k, m] = -1
            A[2 * k - 1, m] = 1
        end
    end
    lb[2] = 0
    ub[2] = 1
    @assert m == M

    return OptProblem(A, c, lb, ub, l, u, vartypes, CycleConstraint[])
end

# TODO: Use Edge objects downstream.
# Investigate whether just returning the iterator is fine.
function get_all_edges(graph)
    return map(e -> (src(e), dst(e)), edges(graph))
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

function solve_IP(O::OptProblem, initial_solution = Int[]; kw...)
    model = LinearQuadraticModel(CbcSolver(;kw...))
    A, lb, ub = add_cycle_constraints_to_formulation(O)
    loadproblem!(model, A, O.l, O.u, O.c, lb, ub, :Max)
    setvartype!(model, O.vartypes)
    original_stdout = stdout
    if !isempty(initial_solution)
        if values(kw).logLevel < 1
            @suppress setwarmstart!(model, initial_solution)
        else
            setwarmstart!(model, initial_solution)
        end
    end
    optimize!(model)
   
    attrs = Dict()
    attrs[:objbound] = getobjbound(model)
    attrs[:solver] = :ip
    solution = MathProgBase.HighLevelInterface.MixintprogSolution(status(model), getobjval(model), getsolution(model), attrs)
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

function follow_path(graph, reverse_edges, w, v1)
    path = [v1]
    while true
        v = path[end]
        successor_found = false
        for v2 in outneighbors(graph, v)
            e = reverse_edges[(v, v2)]
            if w[e]
                w[e] = false
                if v2 == v1
                    return path, true
                end
                push!(path, v2)
                successor_found = true
            end
        end
        if !successor_found
            break
        end
    end

    return path, false
end

function extract_paths(graph, edges, reverse_edges, first_vertex, solution)
    w = solution .>= 1
    n = sum(w)
    main_path, is_cycle = follow_path(graph, reverse_edges, w, first_vertex)
    n -= length(main_path) - 1
    cycles = Vector{Int}[]
    other_paths = Vector{Int}[]
    while n > 0
        if sum(w) != n
            @assert sum(w) == n
        end
        path, is_cycle = follow_path(graph, reverse_edges, w,
                                     edges[findfirst(w)][1])
        if is_cycle
            push!(cycles, path)
        else
            push!(other_paths, path)
        end
        n -= length(path) - !is_cycle
    end

    return main_path, cycles, other_paths
end

mutable struct Cutset
    vertices::Vector{Int}
    total_internal_flow::Float64
    total_external_inflow::Float64
    max_internal_inflow::Float64
end

function Cutset(graph, reverse_edges, vertices::Vector{Int}, w)
    cutset = Cutset(sort(vertices), 0.0, 0.0, 0.0)
    for v in vertices
        internal_inflow = 0.0
        for n in inneighbors(graph, v)
            e = reverse_edges[(n, v)]
            if w[e] > 0
                if n ∈ vertices
                    cutset.total_internal_flow += w[e]
                    internal_inflow += w[e]
                else
                    cutset.total_external_inflow += w[e]
                end
            end
        end
        if internal_inflow > cutset.max_internal_inflow
            cutset.max_internal_inflow = internal_inflow
        end
    end
    return cutset
end

function inflow_change(graph, reverse_edges, cutset::Cutset, vertex, w)
    Δ = 0.0
    for n in outneighbors(graph, vertex)
        e = reverse_edges[(vertex, n)]
        if w[e] > 0
            if n ∈ cutset.vertices
                Δ -= w[e]
            end
        end
    end

    for n in inneighbors(graph, vertex)
        e = reverse_edges[(n, vertex)]
        if w[e] > 0
            if n ∉ cutset.vertices
                Δ += w[e]
            end
        end
    end

    return Δ
end

# TODO: Implement incremental update.
function add_vertex_to_cutset(graph, reverse_edges, cutset::Cutset, vertex, w)
    return Cutset(graph, reverse_edges, vcat(cutset.vertices, vertex), w)
end

function find_fractional_cutsets(graph, edges, reverse_edges, solution,
                                 fragments, first_vertex)
    # TODO: Check whether the copy ends up necessary.
    w = copy(solution)
    candidate_cutsets = Cutset[]
    cutsets = Vector{Int}[]
    fractional_edges = 0 .< w
    for i in findall(fractional_edges)
        if first_vertex ∉ edges[i]
            push!(candidate_cutsets,
                  Cutset(graph, reverse_edges, [edges[i][1], edges[i][2]], w))
        end
    end

    for fragment in sort(fragments, by = length)
        push!(candidate_cutsets, Cutset(graph, reverse_edges, fragment, w))
    end

    while !isempty(candidate_cutsets)
        cutset = popfirst!(candidate_cutsets)
        # TODO: Track cutoff length in a nicer way.
        if !isempty(cutsets) && length(cutset.vertices) > max(12, 2 + length(cutsets[1]))
            break
        end
        margin = cycle_constraints_margin(cutset)
        if margin < -0.001 && cutset.vertices ∉ cutsets
            push!(cutsets, cutset.vertices)
        else
            best_v = -1
            best_inflow_reduction = Inf
            for v in fractional_inneighbors_of_cutset(graph, reverse_edges,
                                                      cutset, fractional_edges,
                                                      first_vertex)
                Δ = inflow_change(graph, reverse_edges, cutset, v, w)
                if Δ < best_inflow_reduction
                    best_v = v
                    best_inflow_reduction = Δ
                end
            end
            if best_v != -1
                push!(candidate_cutsets,
                      add_vertex_to_cutset(graph, reverse_edges, cutset,
                                           best_v, w))
            end
        end
    end

    return cutsets
end

function fractional_inneighbors_of_cutset(graph, reverse_edges, cutset::Cutset,
                                          fractional_edges, first_vertex)
    neighbors = Int[]
    for v in cutset.vertices
        for n in inneighbors(graph, v)
            if n != first_vertex && n ∉ cutset.vertices && fractional_edges[reverse_edges[(n, v)]]
                if n ∉ neighbors
                    push!(neighbors, n)
                end
            end
        end
    end

    return neighbors
end

function cycle_constraints_margin(cutset::Cutset)
    margin = min(length(cutset.vertices) - 1 - cutset.total_internal_flow,
                 cutset.total_external_inflow - cutset.max_internal_inflow)

    return margin
end


function select_cycles(cycles)
    cutoff_length = max(minimum(length.(cycles)), 12)
    return filter(c -> length(c) <= cutoff_length, cycles)
end

# Add constraints derived from cycles in the graph. (Technically it
# doesn't have to be cycles, any arbitrary set of vertices not
# including the starting vertex is fine.)
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
function constrain_cycles!(O::OptProblem, cycles, edges,
                           solution, cycle_constraint_mode)
    previous_number_of_constraints = length(O.cycle_constraints)
    for cycle in cycles
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
            incoming_edges_to_cycle = findall([edge[1] ∉ cycle && edge[2] ∈ cycle for edge in edges])
            # Add one constraint for each vertex in the cycle.
            for vertex in cycle
                A = spzeros(Int, size(O.A, 2))
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
