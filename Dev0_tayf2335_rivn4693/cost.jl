using LinearAlgebra   #   pour la fonction  norm

# truc pour que xref soit une variable non modifiable après la définition
# de la fonction cost: mettre le tout dans un bloc "let".
let xref = [1.0;2.0;3.0]
    global function cost(x)
        f =  0.5 * norm(x - xref)^2
        return f
    end
end

x0 = [1.0;-1.0;1.0]

using NLPModels    # l'outil de modélisation principal que nous étudierons

using ADNLPModels   # pour construire un modèle de programmation non linéaire à partir de la fonction de coût
nlp = ADNLPModel(cost,x0) # AD pour Automatic Différentiation, calcule les dérivées
