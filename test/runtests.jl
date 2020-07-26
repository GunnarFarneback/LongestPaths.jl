using Test
using LongestPaths
using LongestPaths: path_length
using LightGraphs

# Brute force search for the longest path using dfs.
function dfs_longest_path(g::AbstractGraph{T}, weights, first_vertex,
                          last_vertex = 0) where T
    visited = falses(nv(g))
    path = Vector{T}()
    longest_path = Vector{T}()
    push!(path, first_vertex)
    visited[first_vertex] = true
    recurse_dfs_longest_path!(g, weights, last_vertex, visited,
                              path, longest_path)
    return longest_path
end

function recurse_dfs_longest_path!(g, weights, last_vertex, visited,
                                   path, longest_path)
    v = path[end]
    if (last_vertex == 0 || v == last_vertex) && (isempty(longest_path) || path_length(path, weights) > path_length(longest_path, weights))
        resize!(longest_path, length(path))
        copyto!(longest_path, path)
    end
    if v == last_vertex
        return
    end
    for n in outneighbors(g, v)
        if !visited[n]
            push!(path, n)
            visited[n] = true
            recurse_dfs_longest_path!(g, weights, last_vertex, visited, path,
                                      longest_path)
            visited[n] = false
            pop!(path)
        end
    end
end

@testset "path graphs" begin
    for n = 1:10
        g = path_digraph(n)

        r = find_longest_path(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
        @test r.longest_path == 1:n

        r = find_longest_cycle(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) == 0

        for i = 1:n
            r = find_longest_path(g, i, log_level = 0)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - i
            @test r.longest_path == i:n

            r = find_longest_cycle(g, i, log_level = 0)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) == 0
            for j = 1:i-1
                r = find_longest_path(g, i, j, log_level = 0)
                @test r.lower_bound == r.upper_bound == 0
                @test isempty(r.longest_path)
            end

            for j = i+1:n
                r = find_longest_path(g, i, j, log_level = 0)
                @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == j - i
                @test r.longest_path == i:j
            end
        end
    end
end

@testset "cycle graphs" begin
    for n = 2:10
        g = cycle_digraph(n)

        r = find_longest_path(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
        @test r.longest_path == 1:n

        r = find_longest_cycle(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) == n

        for i = 1:n
            r = find_longest_path(g, i, log_level = 0)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
            @test r.longest_path == vcat(i:n, 1:(i-1))
            for j = 1:n
                if i == j
                    r = find_longest_cycle(g, i, log_level = 0)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) == n
                    @test r.longest_path == vcat(i:n, 1:(i-1))
                else
                    r = find_longest_path(g, i, j, log_level = 0)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == mod(j - i, n)
                    if i < j
                        @test r.longest_path == i:j
                    else
                        @test r.longest_path == vcat(i:n, 1:j)
                    end
                end
            end
        end
    end
end

@testset "complete graphs" begin
    for n = 2:10
        g = complete_digraph(n)

        r = find_longest_path(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1

        r = find_longest_cycle(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) == n

        for i = 1:n
            r = find_longest_path(g, i, log_level = 0)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
            for j = 1:n
                if i == j
                    r = find_longest_cycle(g, i, log_level = 0)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) == n
                else
                    r = find_longest_path(g, i, j, log_level = 0)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
                end
            end
        end
    end
end

@testset "bipartite graphs" begin
    for m = 1:10, n = 1:10
        g = DiGraph(complete_bipartite_graph(m, n))
        r = find_longest_path(g, log_level = 0)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == 2 * min(m, n) - (m <= n)
    end        
end

function test_longest_path(g, first_vertex, last_vertex, correct,
                           weights = nothing)
    if first_vertex != last_vertex
        r = find_longest_path(g, first_vertex, last_vertex, log_level = 0,
                              weights = weights, use_ip_warmstart=true)
    else
        r = find_longest_cycle(g, first_vertex, log_level = 0,
                               weights = weights, use_ip_warmstart=true)
    end
    if r.lower_bound ≉ correct || r.upper_bound ≉ correct
        return false
    end
    if !isa(correct, Array)
        return path_length(r.longest_path, r.weights) ≈ correct
    else
        return r.longest_path == correct
    end
end

@testset "sedgewickmaze" begin
    g = DiGraph(smallgraph(:sedgewickmaze))
    all_cycles = simplecycles_hawick_james(g)
    @testset "unweighted" begin
        w = LongestPaths.UnweightedPath()
        longest_paths = dfs_longest_path.((g,), (w,), 1:8, permutedims(1:8))
        @test test_longest_path(g, 0, 0, maximum(length.(all_cycles)))
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(length.(filter(x -> i in x, all_cycles)))
                else
                    if j == 0
                        n = maximum(length.(longest_paths[i, :])) - 1
                    else
                        n = length(longest_paths[i, j]) - 1
                    end
                end
                @test test_longest_path(g, i, j, n)
            end
        end
    end
    @testset "weighted but still equal" begin
        for weights in (Dict(Tuple(e) => 2 for e in edges(g)),
                        Dict(Tuple(e) => 0.1 for e in edges(g)))
            w1 = LongestPaths.WeightedPath(weights)
            w2 = LongestPaths.WeightedCycle(weights)
            longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                              permutedims(1:8))
            @test test_longest_path(g, 0, 0,
                                    maximum(path_length.(all_cycles, (w2,))),
                                    weights)
            for i = 1:8
                for j = 0:8
                    if i == j
                        n = maximum(path_length.(filter(x -> i in x,
                                                        all_cycles), (w2,)))
                        @test test_longest_path(g, i, j, n, weights)
                    else
                        if j == 0
                            n = maximum(path_length.(longest_paths[i, :],
                                                     (w1,)))
                        else
                            n = path_length(longest_paths[i, j], w1)
                        end
                        @test test_longest_path(g, i, j, n, weights)
                    end
                end
            end
        end
    end

    @testset "weighted with integers" begin
        weights = Dict((v1, v2) => (mod(v1, 2) + mod(v2, 3)
                                    + mod(v1 * v2, 5))
                       for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end

    @testset "weighted with floats" begin
        weights = Dict((v1, v2) => (mod(v1, 1.7) + mod(v2, 2.1)
                                    + mod(v1 * v2, 4.9))
                       for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end

    @testset "weighted by -1 (shortest path)" begin
        weights = Dict((v1, v2) => -1 for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end

    @testset "weighted by negative numbers" begin
        weights = Dict((v1, v2) => -(mod(v1, 2) + mod(v2, 3)
                                     + mod(v1 * v2, 5))
                       for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end

    @testset "weighted by mixed signs" begin
        weights = Dict((v1, v2) => (mod(v1, 2) + mod(v2, 3)
                                    - mod(v1 * v2, 5))
                       for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end

    @testset "weighted by zero" begin
        weights = Dict((v1, v2) => 0
                       for (v1, v2) in Tuple.(edges(g)))
        w1 = LongestPaths.WeightedPath(weights)
        w2 = LongestPaths.WeightedCycle(weights)
        longest_paths = dfs_longest_path.((g,), (w1,), 1:8,
                                          permutedims(1:8))
        @test test_longest_path(g, 0, 0,
                                maximum(path_length.(all_cycles, (w2,))),
                                weights)
        for i = 1:8
            for j = 0:8
                if i == j
                    n = maximum(path_length.(filter(x -> i in x,
                                                    all_cycles), (w2,)))
                    @test test_longest_path(g, i, j, n, weights)
                else
                    if j == 0
                        n = maximum(path_length.(longest_paths[i, :],
                                                 (w1,)))
                    else
                        n = path_length(longest_paths[i, j], w1)
                    end
                    @test test_longest_path(g, i, j, n, weights)
                end
            end
        end
    end
end
