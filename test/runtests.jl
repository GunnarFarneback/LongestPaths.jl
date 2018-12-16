using Test
using LongestPaths
using LightGraphs

@testset "Trivial graphs" begin
    for n = 1:100
        r = longest_path(PathDiGraph(n))
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
        r = longest_path(CycleDiGraph(n))
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
    end

    for n = 1:10
        r = longest_path(CompleteDiGraph(n))
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == n - 1
    end

    for m = 1:10, n = 1:10
        r = longest_path(DiGraph(CompleteBipartiteGraph(m, n)))
        @test r.lower_bound == r.upper_bound == length(r.longest_path) - 1 == 2 * min(m, n) - (m <= n)
    end        
end
