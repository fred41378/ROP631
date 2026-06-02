export Dus2_3

fObj(z) = (1.0-(1.0/(5.0*(z^2)-6.0*z+5.0)))


function Dus2_3()
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
