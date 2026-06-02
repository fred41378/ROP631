using NLPModels
using ADNLPModels

function  f31(β :: Real; x0 :: Real = 1.0)

    # les ADNLPModels ont des variables vectorielles, on doit prendre x[1]
    # pour notre fonction scalaire
    func(x) =  1 + x[1]^2 - cos( β * x[1] )

    nlp = ADNLPModel(func, [x0])
    return nlp
end

nlp = f31(5.2)


# graphiques
using Plots
#pyplot()  # mon backend préféré

t = -3.5:0.01:3.5;

f(x) = obj(nlp, [x])
graph_f = plot(t, f.(t), reuse = false)

g(x) = grad(nlp, [x])[1]
graph_g = plot(t, g.(t), reuse = false)



using JSOSolvers

res = lbfgs(nlp, x=[-2.0])   #  optisation; la méthode se nomme lbfgs
x = res.solution
println("x* = ", x, " grad(x) = ",grad(nlp,x))

res = lbfgs(nlp, x=[-1.0])   #  optisation; la méthode se nomme lbfgs
x = res.solution
println("x* = ", x, " grad(x) = ",grad(nlp,x))

res = lbfgs(nlp, x=[0.1])   #  optisation; la méthode se nomme lbfgs
x = res.solution
println("x* = ", x, " grad(x) = ",grad(nlp,x))

res = lbfgs(nlp, x=[1.0])   #  optisation; la méthode se nomme lbfgs
x = res.solution
println("x* = ", x, " grad(x) = ",grad(nlp,x))

res = lbfgs(nlp, x=[2.0])   #  optisation; la méthode se nomme lbfgs
x = res.solution
println("x* = ", x, " grad(x) = ",grad(nlp,x))
# using Logging
# with_logger(NullLogger()) do    #  éviter d'avoir les Info à chaque itération
#     global res = lbfgs(nlp, x=[0.7])
# end
# x = res.solution
# println("x* = ", x, " grad(x) = ",grad(nlp,x))
