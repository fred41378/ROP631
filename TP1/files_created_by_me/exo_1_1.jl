using Optim
using Roots

# Point P
Px = 2.0
Py = 3.0

# Fonction  d(x)
d(x) = (x - Px)^2 + (1/x - Py)^2

# Dérivée d'ordre 1
d1(x) = 2*(x - Px) - (2/x^2)*(1/x - Py)

# Dérivée d'ordre 2
d2(x) = 2 + 6/x^4 - 4*Py/x^3


# On trouve les racines de d'(x) pour trouver les points candidats à l'optimalité.
# Puisque que la solution est difficile à trouver analytiquement, 
# on utilise une méthode numérique pour trouver les racines de d'(x).
racines = find_zeros(d1, 0.01, 10.0)

epsilon = 1e-12

println("=== Résolution avec les points candidats ===\n")
for x_etoile in racines
    println("Point candidat : x* = $x_etoile")
    println("  d'(x*)  = $(d1(x_etoile)) ≈ 0 ? -> $(abs(d1(x_etoile)) < epsilon)")
    println("  d''(x*) = $(d2(x_etoile))")
    if d2(x_etoile) > 0
        println("  => Minimum local (d''(x*) > 0)")
    else
        println("  => Pas un minimum (d''(x*) ≤ 0)")
    end
    println()
end
