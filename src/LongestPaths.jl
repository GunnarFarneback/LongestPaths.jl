"""
    LongestPaths

Find the longest simple path in a directed graph or a long path and a
good upper bound. The longest path problem is NP-hard, so the time
needed to find the solution grows quickly with the size of the graph,
unless it has some advantageous structure.

    longest_path(graph; kwargs)

Find the longest simple path in a LightGraphs directed `graph`,
starting with vertex one and ending anywhere.
"""
module LongestPaths

include("longest_path.jl")
include("utils.jl")

end
