# Algorithme de région de confiance scalaire (unidimensionnel)
#
# La fonction h est approchée par un polynôme de Taylor de degré 2, q.
# Si q approche bien h, alors on se déplace au minimum de q.
# Autremet, on réduit la taille de la région (Δ) jusqu'à ce que le minimum
# de q soit une bonne approximation de h.
# q(d) = h(t) + h'(t) * d + 0.5 * h''(t) * d² 
#
# "key word arguments" 
# eps1 :: Float64. lorsque r est sous cette valeur, on conserve le même x
#                  et réduit Δ
#                  Seuil pour determiner si q est une mauvaise approximation de h
# eps2 :: Float64. lorsque r est sous cette valeur, on sugmente Δ
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
    
    q(d) = fₖ + gₖ * d + 0.5 * H * d^2

    verbose &&
        @printf(" iter  t          gₖ         Δ         pred      ared\n")
    verbose && @printf(" %4d %9.2e  %9.2e  %9.2e \n", iter, t, gₖ, Δ)
    
    # On boucle jusqu'a satisfaire la tolérance prescrite ou bien dépasser le nombre
    # maximum d'itérations
    while ((abs(gₖ) > tol) && (iter < maxiter))
        # Ce code bidon doit être remplacé par l'algorithme de région de confiance.
        # pred = 12.34
        # ared = 24.68
        # iter += 1
        # verbose && @printf(" %4d %9.2e  %9.2e  %9.2e %9.2e %9.2e\n",
        #                    iter, t, gₖ, Δ, pred, ared)

        # choisir le bord de la fonction
        if q(Δ) < q(-Δ)
            dr = Δ
        else
            dr = -Δ
        end

        # calculer le pas de Newton
        dn = -gₖ / H

        # Si le pas de Newton est dans la région de confiance et améliore la fonction, on le choisit
        if (abs(dn) < Δ) && (q(dn) < q(0))
            dr = dn
        end 

        ared = fₖ - obj(h, t + dr)
        pred = q(0) - q(dr)

        r = ared / pred

        if r < eps1
            Δ *= red
        else 
            t += dr
            fₖ = obj(h, t)  
            gₖ = grad(h, t)
            H = hess(h, t)

            q = (d) -> fₖ + gₖ * d + 0.5 * H * d^2

            if r > eps2
                Δ *= aug
            end
        end

        iter += 1
        verbose && @printf(" %4d %9.2e  %9.2e  %9.2e %9.2e %9.2e\n",
                           iter, t, gₖ, Δ, pred, ared)

    end

    optimal = abs(gₖ) <= tol
    tired = iter >= maxiter
    status = optimal ? :first_order : :max_iter
    return (t, fₖ, abs(gₖ), iter, optimal, tired, status)
end
