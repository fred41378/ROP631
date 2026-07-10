

using LinearAlgebra

# Fonction objectif F

function F(x)

    a0 = x[1]
    a3 = x[2]

    return a0^2 + (a0 + a3)^2 + (a0 + 8*a3 - 3)^2 + (a0 + 27*a3 - 9)^2
end

# Gradient

function gradF(x)

    a0 = x[1]
    a3 = x[2]

    return [
        8*a0 + 72*a3 - 24 ;
        72*a0 + 1588*a3 - 534
    ]
end

# Hessienne 

const Q = [
    8.0    72.0
    72.0  1588.0
]

function main()

    # Point initial x0

    x = [0.0 ; 0.0]

    println("================================================")
    println("Gradient conjugué")
    println("================================================")
    println()

    println("x0 = ", x)
    println("Q(x0) = ", F(x))
    println()

    # Initialisation

    q = gradF(x)

    println("q0 = ", q)

    d = -q

    println("d0 = ", d)
    println()

    max_iteration = 10
    tolerence = 1e-10

    for k in 0:max_iteration-1

        println("------------------------------------------------")
        println("Itération ", k + 1)
        println("------------------------------------------------")


        theta =
            -(q' * d)[1] /
            (d' * Q * d)[1]

        println("theta_", k, " = ", theta)

        # Mise à jour

        xnew = x + theta*d

        println("x_", k+1, " = ", xnew)

        # Nouveau gradient

        qnew = gradF(xnew)

        println("q_", k+1, " = ", qnew)

        println("||q|| = ", norm(qnew))

        # Critère d'arrêt

        if norm(qnew) < tolerence

            x = xnew
            q = qnew
            

            println()
            println("Convergence atteinte.")
            break
        end

        beta =
            (qnew' * Q * d)[1] /
            (d' * Q * d)[1]

        println("beta_", k, " = ", beta)

        dnew = -qnew + beta*d

        println("d_", k+1, " = ", dnew)

        println()

        x = xnew
        q = qnew
        d = dnew

    end

    # print des resultats finaux

    println()
    println("================================================")
    println("Résultat final")
    println("================================================")

    println("a0 = ", x[1])
    println("a3 = ", x[2])

    println()

    println("F(x*) = ", F(x))

    println()

    println("Gradient final = ")
    println(gradF(x))

end
main()