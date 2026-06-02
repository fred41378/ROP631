export Duscube

fObj(x)=x^3-(x-4)^2-100*x

function Duscube()
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
