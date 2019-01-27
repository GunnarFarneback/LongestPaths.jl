using Test
using LongestPaths
using LightGraphs

# Brute force search for the longest path using dfs.
function dfs_longest_path(g::AbstractGraph{T}, first_vertex,
                          last_vertex = 0) where T
    visited = falses(nv(g))
    path = Vector{T}()
    longest_path = Vector{T}()
    push!(path, first_vertex)
    visited[first_vertex] = true
    recurse_dfs_longest_path!(g, last_vertex, visited, path, longest_path)
    return longest_path
end

function recurse_dfs_longest_path!(g, last_vertex, visited, path, longest_path)
    v = path[end]
    if (last_vertex == 0 || v == last_vertex) && length(path) > length(longest_path)
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
            recurse_dfs_longest_path!(g, last_vertex, visited, path,
                                      longest_path)
            visited[n] = false
            pop!(path)
        end
    end
end

@testset "path graphs" begin
    for n = 1:10
        g = PathDiGraph(n)

        r = find_longest_path(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
        @test r.longest_path == 1:n

        r = find_longest_cycle(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == -1

        for i = 1:n
            r = find_longest_path(g, i)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - i
            @test r.longest_path == i:n

            r = find_longest_cycle(g, i)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == -1
            for j = 1:i-1
                r = find_longest_path(g, i, j)
                @test r.lower_bound == r.upper_bound == -1
                @test isempty(r.longest_path)
            end

            for j = i+1:n
                r = find_longest_path(g, i, j)
                @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == j - i
                @test r.longest_path == i:j
            end
        end
    end
end

@testset "cycle graphs" begin
    for n = 2:10
        g = CycleDiGraph(n)

        r = find_longest_path(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
        @test r.longest_path == 1:n

        r = find_longest_cycle(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) == n

        for i = 1:n
            r = find_longest_path(g, i)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
            @test r.longest_path == vcat(i:n, 1:(i-1))
            for j = 1:n
                if i == j
                    r = find_longest_cycle(g, i)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) == n
                    @test r.longest_path == vcat(i:n, 1:(i-1))
                else
                    r = find_longest_path(g, i, j)
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

# We need to disable warmstart in some cases to work around
# https://github.com/JuliaOpt/Cbc.jl/issues/99.
@testset "complete graphs" begin
    for n = 2:10
        g = CompleteDiGraph(n)

        r = find_longest_path(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1

        r = find_longest_cycle(g, use_ip_warmstart = false)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) == n

        for i = 1:n
            r = find_longest_path(g, i)
            @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
            for j = 1:n
                if i == j
                    r = find_longest_cycle(g, i, use_ip_warmstart = false)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) == n
                else
                    r = find_longest_path(g, i, j)
                    @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
                end
            end
        end
    end
end

@testset "bipartite graphs" begin
    for m = 1:10, n = 1:10
        g = DiGraph(CompleteBipartiteGraph(m, n))
        r = find_longest_path(g)
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == 2 * min(m, n) - (m <= n)
    end        
end

# We need to disable warmstart to work around
# https://github.com/JuliaOpt/Cbc.jl/issues/94.
function test_longest_path(g, first_vertex, last_vertex, correct)
    if first_vertex != last_vertex
        r = find_longest_path(g, first_vertex, last_vertex,
                              use_ip_warmstart = false)
    else
        r = find_longest_cycle(g, first_vertex, use_ip_warmstart = false)
    end
    if r.lower_bound != correct || r.upper_bound != correct
        return false
    end
    if !isa(correct, Array)
        if length(r.longest_path) - !r.is_cycle != correct
            return false
        end
    else
        if r.longest_path != correct
            return false
        end
    end
    return true
end

@testset "sedgewickmaze" begin
    g = DiGraph(smallgraph(:sedgewickmaze))
    all_cycles = simplecycles_hawick_james(g)
    longest_paths = dfs_longest_path.((g,), 1:8, permutedims(1:8))
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
