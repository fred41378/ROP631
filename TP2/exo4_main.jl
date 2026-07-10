using LinearAlgebra
using Printf
using Plots

# Pas d'Armijo
function pas_admissible_armijo(x, d, f, ∇f, tow0)
    g = ∇f(x)
    gd = dot(g,d)
    θ = 1.0
    while f(x .+ θ * d) - f(x) > θ * tow0 * gd
        θ /= 2
    end
    return θ
end

# Descente de gradient avec pas d'Armijo
function descente_armijo(f, ∇f, x0, tow0=0.5)
    x = copy(x0)
    res_x = [copy(x)]
    res_f = [f(x)]
    res_g = [norm(∇f(x))]
    res_iter = [0]
    tolerence = 1e-8
    maxiter = 3000
    while res_g[end] > tolerence && length(res_x) < maxiter
        g = ∇f(x)
        d = -g
        theta = pas_admissible_armijo(x, d, f, ∇f, tow0)
        x = x + theta * d
        push!(res_x, copy(x))
        push!(res_f, f(x))
        push!(res_g, norm(∇f(x)))
        push!(res_iter, length(res_x) - 1)
    end
    return res_x, res_f, res_g, res_iter
end 

# Fonction pour calculer alpha α
function get_alpha(δ, p, ∆)
    a = dot(p, p)
    if a < 1e-28
        return 0.0
    end
    b = 2 * dot(δ, p)
    c = dot(δ, δ) - ∆^2
    disc = b^2 - 4*a*c
    disc = max(disc, 0.0)
    return (-b + sqrt(disc)) / (2*a)
end

# Algorithme de la région de confiance
function conjugue(c, Q, ∆)
    ξ = 1.0
    ϵ = 1e-8
    δ = zeros(length(c))
    ∇q = Q*δ .+ c
    norm∇q0 = norm(∇q)
    p = -∇q
    maxiter = 200
    sortie = false
    assez_precis = false
    α = 0.0
    for k in 1:maxiter
        pQp = dot(p, Q*p)
        α = get_alpha(δ, p, ∆)
        sortie = pQp <= 0
        assez_precis = norm(∇q) < min(ϵ, norm∇q0^(1+ξ))

        if !sortie
            ϑ = -dot(∇q, p) / pQp
            sortie = ϑ > α
            if !sortie
                δ = δ .+ ϑ .* p
                ∇q_new = Q*δ .+ c
                β = dot(∇q_new, Q*p) / pQp
                p = -∇q_new .+ β .* p
                ∇q = ∇q_new
            end            
        end

        if sortie || assez_precis
            break
        end
    end

        if sortie
            δ = δ .+ α .* p
        end

    return δ
end

# Algorithme de Newton avec région de confiance (Version du Gradient Conjugué)
function newton_region_confiance(f, ∇f, ∇²f, x0, ∆0=0.5; ϵ=1e-8, ξ=1.0)
    x = copy(x0)
    ∆ = ∆0
    res_x = [copy(x)]
    res_f = [f(x)]
    res_g = [norm(∇f(x))]
    res_iter = [0]
    maxiter = 200
    for k in 1:maxiter
        c = ∇f(x)
        Q = ∇²f(x)
        dr = conjugue(c, Q, ∆)

        # Parfois, la direction de descente est trop petite, on sort
        # pour éviter les problèmes numériques, on sort si la norme est trop petite ou si on a un NaN
        # L'algo fonctionne sans, mais c'est plus propre de sortir dans ce cas
        if norm(dr) < 1e-14
            break
        end

        q_dr = 0.5 * dot(dr, Q*dr) + dot(c, dr)
        r = (f(x) - f(x .+ dr)) / (0 - q_dr)
        if r < 0.25
            ∆ /= 2
        else
            x = x .+ dr  
        end
        if r > 0.75 && norm(dr) == ∆
            ∆ *= 2
        end

        push!(res_x, copy(x))
        push!(res_f, f(x))
        push!(res_g, norm(∇f(x)))
        push!(res_iter, length(res_x) - 1)
        if norm(∇f(x)) < ϵ
            break
        end

    end
    return res_x, res_f, res_g, res_iter
end

# f(x) = (x1^2+x2-11)^2 + (x1+x2^2-7)^2
function f1(x)
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    return a^2 + b^2
end

# Gradient de f1
function grad_f1(x)
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    g1 = 4*x[1]*a + 2*b
    g2 = 2*a + 4*x[2]*b
    return [g1, g2]
end

# Hessienne de f1
function hess_f1(x)
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    h11 = 12*x[1]^2 + 4*x[2] - 42
    h12 = 4*x[1] + 4*x[2]
    h22 = 12*x[2]^2 + 4*x[1] - 26
    return [h11 h12; h12 h22]
end

# f(x) = 100(x2-x1^2)^2 + (1-x1)^2
# Rosenbrock
function f2(x)
    return 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
end

# Gradient de f2
function grad_f2(x)
    g1 = -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1])
    g2 = 200*(x[2]-x[1]^2)
    return [g1, g2]
end

# Hessienne de f2
function hess_f2(x)
    h11 = 1200*x[1]^2 - 400*x[2] + 2
    h12 = -400*x[1]
    h22 = 200.0
    return [h11 h12; h12 h22]
end

function afficher_tableau(nom, res_f, res_g)
    println("================================================")
    println(nom)
    println("================================================")
    println()
    println(rpad("k", 5), rpad("f(x_k)", 20), "||grad f(x_k)||")
    n = length(res_f)
    # Pour éviter d'afficher trop de lignes, on affiche les 8 premières et les 8 dernières si n > 15
    idx = n <= 15 ? (1:n) : vcat(1:8, (n-6):n)
    for k in idx
        # Formattage du tableau avec le formating de Printf
        @printf("%-5d%-20.10g%-20.10g\n", k-1, res_f[k], res_g[k])
    end
    println("Nombre d'itérations = ", n-1)
end


# Fonction principale avec l'affichage des résultats pour les deux fonctions f1 et f2
function main()      # point de départ pour f1 (arbitraire)
    x0 = [-1.2, 1.0]       # point de départ imposé pour Rosenbrock

    println("############################################################")
    println("# Fonction 1 : (x1^2+x2-11)^2 + (x1+x2^2-7)^2")
    println("############################################################")

    axf1, af1, ag1, aif1 = descente_armijo(f1, grad_f1, x0)
    afficher_tableau("Armijo -- f1", af1, ag1)

    rxf1, rf1, rg1, rif1 = newton_region_confiance(f1, grad_f1, hess_f1, x0)
    afficher_tableau("Region de confiance (GC) -- f1", rf1, rg1)

    println("\n############################################################")
    println("# Fonction 2 (Rosenbrock) : x0 = (-1.2, 1.0)")
    println("############################################################")

    axf2, af2, ag2, aif2 = descente_armijo(f2, grad_f2, x0)
    afficher_tableau("Armijo -- Rosenbrock", af2, ag2)

    rxf2, rf2, rg2, rif2 = newton_region_confiance(f2, grad_f2, hess_f2, x0)
    afficher_tableau("Region de confiance (GC) -- Rosenbrock", rf2, rg2)


    pf1_arm = plot(aif1, log.(af1), label="Armijo", ylabel = "log(f1(x))", lc=:blue)
    pf1_rc = plot(rif1, log.(rf1), label="Region de confiance", ylabel = "log(f1(x))", lc=:red)
    pgf1_arm = plot(aif1, log.(ag1), label="Armijo", ylabel = "||grad f1(x)||", lc=:blue)
    pgf1_rc = plot(rif1, log.(rg1), label="Region de confiance", ylabel = "||grad f1(x)||", lc=:red)
    plot(pf1_arm, pf1_rc, pgf1_arm, pgf1_rc, layout=(2, 2), title="Convergence des fonctions")

    pf2_arm = plot(aif2, log.(af2), label="Armijo", ylabel = "log(f2(x))", lc=:blue)
    pf2_rc = plot(rif2, log.(rf2), label="Region de confiance", ylabel = "log(f2(x))", lc=:red)
    pgf2_arm = plot(aif2, log.(ag2), label="Armijo", ylabel = "||grad f2(x)||", lc=:blue)
    pgf2_rc = plot(rif2, log.(rg2), label="Region de confiance", ylabel = "||grad f2(x)||", lc=:red)
    plot(pf2_arm, pf2_rc, pgf2_arm, pgf2_rc, layout=(2, 2), title="Convergence des fonctions")
end

main()