export save_path, load_path

# Write `ceil(log2(max_value))` bits representing `value - 1` into
# `buffer`, where `value` is required to be in the range
# `1:max_value`.
function write_bits!(buffer, value, max_value)
    @assert 1 <= value <= max_value
    max_value -= 1
    value -= 1
    while max_value > 0
        push!(buffer, Bool(value & 1))
        value >>= 1
        max_value >>= 1
    end
end

# Opposite of `write_bits!`.
function read_bits!(buffer, max_value)
    @assert 1 <= max_value
    max_value -= 1
    bit = 1
    value = 0
    while max_value > 0
        if popfirst!(buffer)
            value |= bit
        end
        bit <<= 1
        max_value >>= 1
    end
    return value + 1
end

function save_path(filename, graph, path)
    bits = BitVector(undef, 0)
    write_bits!(bits, length(path), nv(graph) + 1)
    write_bits!(bits, path[1], maximum(vertices(graph)))

    visited = falses(nv(graph))
    for i = 1:length(path) - 1
        v1, v2 = path[i], path[i + 1]
        visited[v1] = true
        neighbors = filter(x -> !visited[x], outneighbors(graph, v1))
        i = findfirst(isequal(v2), sort(neighbors))
        @assert i != nothing
        write_bits!(bits, i, length(neighbors))
    end

    buffer = IOBuffer()
    write(buffer, bits)
    write(filename, reinterpret(UInt8, take!(buffer))[1:ceil(Int, length(bits) / 8)])
end

function load_path(filename, graph)
    bytes = read(filename)
    bits = BitVector(undef, 8 * length(bytes))
    while mod(length(bytes), 8) != 0
        push!(bytes, 0x00)
    end
    read!(IOBuffer(bytes), bits)

    n = read_bits!(bits, nv(graph) + 1)
    path = Vector{eltype(graph)}(undef, 0)
    push!(path, read_bits!(bits, maximum(vertices(graph))))

    visited = falses(nv(graph))
    for i = 1:(n - 1)
        v1 = path[end]
        visited[v1] = true
        neighbors = filter(x -> !visited[x], outneighbors(graph, v1))
        v2 = sort(neighbors)[read_bits!(bits, length(neighbors))]
        push!(path, v2)
    end

    return path
end
