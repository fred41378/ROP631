import Pkg
Pkg.activate("ROP631")

include("cost.jl")
let  #  pour obliger f et g à être des variables locales...
    f = obj(nlp, zeros(3))
    println(" f = ", f)  #  devrait être  (norm(xref)^2)/2 = 7
    g = grad(nlp,zeros(3)) 
    println(" g = ", g) #  devrait être  -xref' = [-1.0 -2.0 -3.0]
end

include("exA.3.1.jl")
include("exA.3.2.jl")