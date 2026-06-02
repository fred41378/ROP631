function  f(x)
    # exemple de fonction f de l'exercice A.3.2
    Q=[1.0 0.5;0.5 3.0]
    q=x'*Q*x  +  0.1
    return log(q)
end

# construit une fonction F avec une cible p
function buildF(p::Vector, f::Function)
    F(x) = [x;f(x)] - p
    return F
end

F = buildF([1.0; 2.0; 3.0], f)

            
function ϕ(x)
    # fonction phi de l'exercice A.3.2
    return 0.5*F(x)'*F(x)
end

using NLPModels
using ForwardDiff

# modèle ADNLSModel de ϕ (AD pour Automatic Differentiation et NLS pour Nonlinear Least Squares)
ADNLSϕ = ADNLSModel(F, zeros(2), 3)

# modèle ADNLPModel de ϕ (AD pour Automatic Differentiation et NLP pour Non Linear Problem )
ADNLPϕ = ADNLPModel(ϕ, zeros(2))



# graphes des deux fonctions
using Plots
#pyplot()  # backend que je préfère

n = 50;
xgrid = range(-3, stop=3, length=n)
ygrid = xgrid
z = zeros(n, n)

z = f.([vec([xi yi]) for xi in xgrid, yi in ygrid])
fig1 = surface(xgrid, ygrid, z, α=0.7, reuse = false)


z = ϕ.([vec([xi yi]) for xi in xgrid, yi in ygrid]);
fig2 = surface(xgrid, ygrid, z, α=0.7, reuse = false)

# Une fois le fichier exécuté, tapez "fig1" et "fig2" dans le REPL pour afficher les deux graphes.



# Résolution du problème d'optimisation avec chacun des 2 modèles NLS et NLP
using JSOSolvers

res = lbfgs(ADNLSϕ)
x = res.solution
println("x* = ", x, "  grad(x) = ",grad(ADNLSϕ,x))

res = lbfgs(ADNLPϕ)
x = res.solution
println("x* = ", x, "  grad(x) = ",grad(ADNLPϕ,x))


#####################################   Exercice A.3.2 d)
# Construisons gradϕ

gradf = x ->  ForwardDiff.gradient(f, x)# Mettre le bon calcul pour ∇f


# Retourne la matrice jacobienne de la fonction F(x)
function jacF(x::Vector)
    J = zeros(3,2)
    jacF!(x, J)
    return J
end

# Par convention, une fonction qui modifie un paramètre (ici, on met le résultat dans J)
# se termine par "!"

# Retourne la jacobienne de F(x) dans une matrice pré-allouée J passée en paramètre
function jacF!(x::Vector, J::Matrix)
    # Mettre le bon calcul
    # L'affectation J[:] dépose le résultat dans la matrice reçue en paramètre et déjà allouée.
    # le "!" est une convention pour signaler que la fonction modifie un de ses paramètres.
    J[:] = ForwardDiff.jacobian(F, x)
    return J
end

# Retourne le gradient de ϕ(x)
function gradϕ(x :: Vector)
    v = similar(x)
    gradϕ!(x, v)
    return v
end

# Retourne le gradient de ϕ(x) dans un vecteur pré-alloué passé en paramètre
function gradϕ!(x::Vector, v::Vector)
    # calculer le gradient
    # Mettre le bon calcul dans le vecteur v pré-alloué.
    v[:] = ForwardDiff.gradient(ϕ, x)
    return v
end

####################################### 
println("\n\nComparer votre calcul de gradϕ avec celui des ADNLPModel et ADNLSModel.")
println("x* = ", x, "  gradADNLP(x) = ",grad(ADNLPϕ,x))
println("x* = ", x, "  gradADNLS(x) = ",grad(ADNLSϕ,x))
println("x* = ", x, "  gradϕ(x) = ",gradϕ(x))
