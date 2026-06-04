# Algorithme de région de confiance scalaire (unidimensionnel)
#
# La fonction h est approchée par un polynôme de Taylor de degré 2, q.
# Si q approche bien h, alors on se déplace au minimum de q.
# Autrement, on réduit la taille de la région (Δ) jusqu'à ce que le minimum
# de q soit une bonne approximation de h.
# q(d) = h(t) + h'(t) * d + 0.5 * h''(t) * d² 
#
# "key word arguments" 
# eps1 :: Float64. lorsque r est sous cette valeur, on conserve le même x
#                  et réduit Δ
#                  Seuil pour determiner si q est une mauvaise approximation de h
# eps2 :: Float64. lorsque r est sous cette valeur, on augmente Δ
#                  Seuil pour determiner si q est une très bonne approximation de h
# red :: Float64.  taux de réduction de Δ
# aug :: Float64.  taux d'augmentation de Δ
# Δ :: Float64.    taille initiale de Δ

function TR_1D(h :: LineModel,
               t₀ :: Float64;
               tol :: Float64 = 1e-7,
               maxiter :: Int = 50,
               verbose :: Bool = true,
               eps1 :: Float64 = 0.2,
               eps2 :: Float64 = 0.8,
               red:: Float64 = 0.5,
               aug :: Float64 = 2.0,
               Δ :: Float64 = 1.0)
    
    t = t₀; iter = 0; # Point de départ et compteur d'itérations
    fₖ = obj(h, t)
    gₖ = grad(h, t)
    H = hess(h, t)
    
    verbose &&
        @printf(" iter  t          gₖ         Δ         pred      ared\n")
    verbose && @printf(" %4d %9.2e  %9.2e  %9.2e \n", iter, t, gₖ, Δ)
    
    # On boucle jusqu'a satisfaire la tolérance prescrite ou bien dépasser le nombre
    # maximum d'itérations
    while ((abs(gₖ) > tol) && (iter < maxiter))
        
        q(d) = fₖ + gₖ * d + 0.5 * H * d^2
        if (q(Δ) < q(-Δ)) 
            d_R = Δ 
        else 
            d_R = -Δ
        end
        d_N = - gₖ/H
        if (abs(d_N) < Δ && q(d_R) > q(d_N))
            d_R = d_N
        end
        ared = fₖ - obj(h, t+d_R)
        pred = q(0) - q(d_R)
        r = ared/pred
        if (r < eps1)
            Δ = red * Δ
        else
            t = t + d_R
            #recalculer les valeurs de la fonction objectif et dérivées
            fₖ = obj(h, t)
            gₖ = grad(h, t)
            H = hess(h, t)
        end
        if (r > eps2)
            Δ = aug * Δ
        end


        iter += 1
        verbose && @printf(" %4d %9.2e  %9.2e  %9.2e %9.2e %9.2e\n",
                           iter, t, gₖ, Δ, pred, ared)
    end

    #on stocke la solution dans une nouvelle variable et on repart les itérations à partir du t initial
    t_bar = t
    t = t₀
    iter = 0
    fₖ = obj(h, t)
    gₖ = grad(h, t)
    H = hess(h, t)
    erreur = t - t_bar
    Δ = 1.0
    pred = ared = NaN
    
    verbose &&
        @printf(" iter  t          gₖ         Δ         pred      ared      erreur\n")
    verbose && @printf(" %4d %9.2e  %9.2e  %9.2e %9.2e %9.2e %9.2e \n", iter, t, gₖ, Δ, pred, ared, erreur)
    

    #on refait la boucle ayant en main la solution t pour calculer l'erreur à chaque itération
    while ((abs(gₖ) > tol) && (iter < maxiter))
        
        q(d) = fₖ + gₖ * d + 0.5 * H * d^2
        if (q(Δ) < q(-Δ)) 
            d_R = Δ 
        else 
            d_R = -Δ
        end
        d_N = - gₖ/H
        if (abs(d_N) < Δ && q(d_R) > q(d_N))
            d_R = d_N
        end
        ared = fₖ - obj(h, t+d_R)
        pred = q(0) - q(d_R)
        r = ared/pred
        if (r < eps1)
            Δ = red * Δ
        else
            t = t + d_R
            fₖ = obj(h, t)
            gₖ = grad(h, t)
            H = hess(h, t)
        end
        if (r > eps2)
            Δ = aug * Δ
        end

        erreur = t - t_bar

        iter += 1
        verbose && @printf(" %4d %9.2e  %9.2e  %9.2e %9.2e %9.2e %9.2e\n",
                           iter, t, gₖ, Δ, pred, ared, erreur)
    end

    optimal = ifelse(abs(gₖ) <= tol, true, false)
    tired = ifelse(iter >= maxiter, true, false)
    status = ifelse(optimal && !tired, :Optimal, :NotSolved)
    return (t, fₖ, abs(gₖ), iter, optimal, tired, status)
end
