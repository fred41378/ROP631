# JuMP et NLPModels sont des outils de modélisation.

using JuMP, NLPModels, NLPModelsJuMP
using SolverTools  #  pour avoir le LineModel

using Printf
include("TR_1D_fr.jl")


# Exemple de la table 2.3 des notes; la figure 2.8 illustre la fonction
#
# Quelques trucs julia pour ne spécifier le nom du test une seule fois...

name = "Duscube"
println(" Exemple avec le test ",name)
symb = Symbol(name)
file = string(name, ".jl")

include(file)
f = getfield(Main, symb)

nlp = MathOptNLPModel(f())

h = LineModel(nlp, [0.0], [1.0]);

# Enlever le maxiter = 1 lorsque le TR_1D fonctionnera
(t, f, absg, iter, optimal, tired, status) = TR_1D(h, 2.0, verbose = true)

println(" ------- ")
@printf("Résultat du test: \n t*       =  %9.3e\n f(t*)    =  %9.3e\n |f'(t*)| =  %9.3e\n",t,f,absg)
@printf("\n #iterations  = %2d\n #eval\n   obj   = %2d\n   grad  = %2d\n   hess  = %2d\n", iter, h.counters.neval_obj, h.counters.neval_grad, h.counters.neval_hess)
@printf("\n status = %s \n\n", status)
