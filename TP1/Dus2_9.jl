export Dus2_9

fObj(x)= 1-12*x+7.5*x^2-x^3

function Dus2_9()
    nlp = Model()
    
    @variable(nlp, x, start=0.0)
    
    JuMP.register(nlp, :fObj, 1, fObj, autodiff=true)
    
    @NLobjective(
        nlp,
        Min,
        fObj(x)
    )
    
    return nlp
end
