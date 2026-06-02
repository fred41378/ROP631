export Dus2_1

fObj(x) = exp(x * (x-1))

function Dus2_1()
    nlp = Model()
    
    JuMP.register(nlp, :fObj, 1, fObj, autodiff=true)
    
    
    @variable(nlp, x, start=0.0)
    
    @NLobjective(
        nlp,
        Min,
        fObj(x)
    )
    
    return nlp
end
