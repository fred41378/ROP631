using LinearAlgebra
using Printf

function pas_admissible_armijo(x, d, f, ∇f, tow0)
    g = ∇f(x)
    gd = dot(g,d)
    θ = 1.0
    while f(x .+ θ * d) - f(x) > θ * tow0 * gd
        θ /= 2
    end
    return θ
end


function descente_armijo(f, ∇f, x0, tow0=0.5)
    x = copy(x0)
    res_x = [copy(x)]
    res_f = [f(x)]
    res_g = [norm(∇f(x))]
    tolerence = 1e-8
    while res_g[end] > tolerence
        g = ∇f(x)
        d = -g
        theta = pas_admissible_armijo(x, d, f, ∇f, tow0)
        x = x + theta * d
        push!(res_x, copy(x))
        push!(res_f, f(x))
        push!(res_g, norm(∇f(x)))
    end
    return res_x, res_f, res_g
end 



function get_alpha(δ, p, ∆)
    a = dot(p, p)
    if a < 1e-28
        return Inf
    end
    b = 2 * dot(δ, p)
    c = dot(δ, δ) - ∆^2
    disc = b^2 - 4*a*c
    disc = max(disc, 0.0)
    return (-b + sqrt(disc)) / (2*a)
end

function region_confiance(c, Q, ∆)
    ξ = 1.0
    ϵ = 1e-10
    δ = zeros(length(c))
    ∇q = copy(c)
    norm∇q0 = norm(∇q)
    p = -∇q
    maxiter = 200
    sortie = false
    assez_precis = false
    α = 0.0
    for _ in 1:maxiter
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



function newton_region_confiance(f, ∇f, ∇²f, x0, ∆0=1.0; ϵ=1e-10, ξ=1.0)
    x = copy(x0)
    ∆ = ∆0
    res_x = [copy(x)]
    res_f = [f(x)]
    res_g = [norm(∇f(x))]
    maxiter = 200
    for _ in 1:maxiter
        c = ∇f(x)
        Q = ∇²f(x)
        dr = region_confiance(c, Q, ∆)

        if norm(dr) < 1e-14 || isnan(dr[1])
            break
        end

        q_dr = 0.5 * dot(dr, Q*dr) + dot(c, dr)
        r = (f(x) - f(x .+ dr)) / (0 - q_dr)
        if r < 0.25
            ∆ /= 4
        else
            x = x .+ dr  
        end
        if r > 0.75 && norm(dr) == ∆
            ∆ = min(2*∆, 100.0)
        end

        push!(res_x, copy(x))
        push!(res_f, f(x))
        push!(res_g, norm(∇f(x)))
        
        if norm(∇f(x)) < ϵ
            break
        end

    end
    return res_x, res_f, res_g
end

# f(x) = (x1^2+x2-11)^2 + (x1+x2^2-7)^2
function f1(x::Vector{Float64})
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    return a^2 + b^2
end

function grad_f1(x::Vector{Float64})
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    g1 = 4*x[1]*a + 2*b
    g2 = 2*a + 4*x[2]*b
    return [g1, g2]
end

function hess_f1(x::Vector{Float64})
    a = x[1]^2 + x[2] - 11
    b = x[1] + x[2]^2 - 7
    h11 = 12*x[1]^2 + 4*x[2] - 42
    h12 = 4*x[1] + 4*x[2]
    h22 = 12*x[2]^2 + 4*x[1] - 26
    return [h11 h12; h12 h22]
end

# f(x) = 100(x2-x1^2)^2 + (1-x1)^2
function f2(x::Vector{Float64})
    return 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
end

function grad_f2(x::Vector{Float64})
    g1 = -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1])
    g2 = 200*(x[2]-x[1]^2)
    return [g1, g2]
end

function hess_f2(x::Vector{Float64})
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
    idx = n <= 15 ? (1:n) : vcat(1:8, (n-6):n)
    for k in idx
        @printf("%-5d%-20.10g%-20.10g\n", k-1, res_f[k], res_g[k])
    end
    println("Nombre d'itérations = ", n-1)
end

function main()
    x0_1 = [0.0, 0.0]        # point de départ pour f1 (arbitraire)
    x0_2 = [-1.2, 1.0]       # point de départ imposé pour Rosenbrock

    println("############################################################")
    println("# Fonction 1 : (x1^2+x2-11)^2 + (x1+x2^2-7)^2")
    println("############################################################")

    axf, af, ag = descente_armijo(f1, grad_f1, x0_1)
    afficher_tableau("Armijo -- f1", af, ag)

    rxf, rf, rg = newton_region_confiance(f1, grad_f1, hess_f1, x0_1)
    afficher_tableau("Region de confiance (GC) -- f1", rf, rg)

    println("\n############################################################")
    println("# Fonction 2 : Rosenbrock, x0 = (-1.2, 1.0)")
    println("############################################################")

    axf2, af2, ag2 = descente_armijo(f2, grad_f2, x0_2)
    afficher_tableau("Armijo -- Rosenbrock", af2, ag2)

    rxf2, rf2, rg2 = newton_region_confiance(f2, grad_f2, hess_f2, x0_2)
    afficher_tableau("Region de confiance (GC) -- Rosenbrock", rf2, rg2)
end

main()