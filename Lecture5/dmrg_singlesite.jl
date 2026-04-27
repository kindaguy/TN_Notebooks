using ITensors

# ─────────────────────────────────────────────
# 1. Costruzione dell'MPO per la catena XX
# ─────────────────────────────────────────────
function xx_mpo(sites; J=1.0)
    N = length(sites)
    ampo = AutoMPO()
    for j in 1:N-1
        ampo += J, "S+", j, "S-", j+1
        ampo += J, "S-", j, "S+", j+1
    end
    return MPO(ampo, sites)
end

# ─────────────────────────────────────────────
# 2. Blocchi ambientali L e R
#    Costruisce i proiettori sinistro/destro
#    da usare nella contrazione dell'Heff
# ─────────────────────────────────────────────

"""
    build_left_blocks(psi, H) -> Vector{ITensor}

Calcola i blocchi sinistri L[j] per j = 1..N-1.
L[j] è la contrazione di bra, MPO e ket sui siti 1..j.
"""
function build_left_blocks(psi::MPS, H::MPO)
    N = length(psi)
    L = Vector{ITensor}(undef, N)

    # Blocco iniziale: scalare 1
    L[1] = psi[1] * H[1] * dag(prime(psi[1]))

    for j in 2:N-1
        L[j] = L[j-1] * psi[j] * H[j] * dag(prime(psi[j]))
    end
    return L
end

"""
    build_right_blocks(psi, H) -> Vector{ITensor}

Calcola i blocchi destri R[j] per j = 2..N.
R[j] è la contrazione di bra, MPO e ket sui siti j..N.
"""
function build_right_blocks(psi::MPS, H::MPO)
    N = length(psi)
    R = Vector{ITensor}(undef, N)

    R[N] = psi[N] * H[N] * dag(prime(psi[N]))

    for j in N-1:-1:2
        R[j] = R[j+1] * psi[j] * H[j] * dag(prime(psi[j]))
    end
    return R
end

# ─────────────────────────────────────────────
# 3. Hamiltoniana effettiva al sito k
# ─────────────────────────────────────────────

"""
    heff_apply(v, k, H, L, R) -> ITensor

Applica H_eff al tensore v al sito k.
H_eff = L[k-1] * H[k] * R[k+1]
"""
function heff_apply(v::ITensor, k::Int, H::MPO,
                    L::Vector{ITensor}, R::Vector{ITensor})
    N = length(H)
    Hv = v * H[k]
    if k > 1
        Hv = Hv * L[k-1]
    end
    if k < N
        Hv = Hv * R[k+1]
    end
    return noprime(Hv)
end

# ─────────────────────────────────────────────
# 4. Iterazione di Davidson (Lanczos semplificato)
#    Risolve il problema agli autovalori effettivo
# ─────────────────────────────────────────────

"""
    davidson(apply_H, v0; maxiter, tol) -> (E, v)

Metodo di Davidson per trovare il ground state di H_eff.
apply_H: funzione che applica H_eff al tensore v.
"""
function davidson(apply_H, v0::ITensor; maxiter=10, tol=1e-10)
    # Rappresentiamo lo spazio di Krylov
    V = [v0 / norm(v0)]
    Hv = apply_H(V[1])

    E_old = real(scalar(dag(V[1]) * Hv))
    E = E_old
    v = V[1]

    for iter in 1:maxiter
        # Residuo
        r = Hv - E * V[end]
        res_norm = norm(r)

        if res_norm < tol
            break
        end

        # Ortogonalizza il residuo rispetto allo spazio corrente
        for vi in V
            r = r - scalar(dag(vi) * r) * vi
        end
        r_norm = norm(r)
        if r_norm < 1e-14
            break
        end
        push!(V, r / r_norm)

        # Costruisci la matrice di proiezione
        dim = length(V)
        M = zeros(ComplexF64, dim, dim)
        HV = [apply_H(vi) for vi in V]
        for i in 1:dim, j in 1:dim
            M[i,j] = scalar(dag(V[i]) * HV[j])
        end
        M = (M + M') / 2  # Hermitianizza

        # Diagonalizza
        evals, evecs = eigen(M)
        idx = argmin(real.(evals))
        E = real(evals[idx])
        c = evecs[:, idx]

        # Aggiorna il vettore soluzione
        v = sum(c[i] * V[i] for i in 1:dim)
        v = v / norm(v)
        Hv = apply_H(v)

        if abs(E - E_old) < tol
            break
        end
        E_old = E
    end

    return E, v
end

# ─────────────────────────────────────────────
# 5. DMRG single-site
# ─────────────────────────────────────────────

"""
    dmrg_singlesite(H, psi0; nsweeps, maxdim, cutoff, tol_davidson)

Algoritmo DMRG single-site.

# Parametri
- `H`: MPO dell'Hamiltoniana
- `psi0`: MPS iniziale
- `nsweeps`: numero di sweep
- `maxdim`: bond dimension massima
- `cutoff`: cutoff SVD
- `tol_davidson`: tolleranza per il solutore di Davidson

# Ritorna
- `E`: energia del ground state
- `psi`: MPS ottimizzato
"""
function dmrg_singlesite(H::MPO, psi0::MPS;
                         nsweeps::Int=10,
                         maxdim::Int=100,
                         cutoff::Float64=1e-10,
                         tol_davidson::Float64=1e-10)

    N = length(H)
    psi = copy(psi0)

    # Porta psi in forma canonica sinistra
    orthogonalize!(psi, 1)

    E = 0.0

    for sweep in 1:nsweeps

        # ── Sweep da sinistra a destra ──────────────────
        # Ricostruisce i blocchi all'inizio di ogni sweep
        R = build_right_blocks(psi, H)
        L = Vector{ITensor}(undef, N)  # L[k] viene riempito on-the-fly

        for k in 1:N-1
            # Blocco sinistro al sito k
            if k == 1
                L_k = nothing
            else
                L_k = L[k-1]
            end

            # Funzione che applica H_eff al sito k
            apply_H = v -> begin
                Hv = v * H[k]
                if k > 1; Hv = Hv * L[k-1]; end
                Hv = Hv * R[k+1]
                noprime(Hv)
            end

            # Ottimizza il tensore al sito k
            E, psi[k] = davidson(apply_H, psi[k];
                                  maxiter=20, tol=tol_davidson)

            # SVD e aggiorna: porta l'ortogonalità al sito k+1
            linds = (k == 1) ? [siteind(psi, k)] :
                               [linkind(psi, k-1), siteind(psi, k)]
            U, S, V = svd(psi[k], linds;
                          maxdim=maxdim, cutoff=cutoff,
                          lefttags="Link,l=$k")
            psi[k] = U
            psi[k+1] = S * V * psi[k+1]

            # Aggiorna il blocco sinistro
            L[k] = (k == 1) ?
                psi[k] * H[k] * dag(prime(psi[k])) :
                L[k-1] * psi[k] * H[k] * dag(prime(psi[k]))
        end

        # ── Sweep da destra a sinistra ──────────────────
        # Ricostruisce R on-the-fly
        R = Vector{ITensor}(undef, N)

        for k in N:-1:2
            apply_H = v -> begin
                Hv = v * H[k]
                if k > 1; Hv = Hv * L[k-1]; end
                if k < N; Hv = Hv * R[k+1]; end
                noprime(Hv)
            end

            E, psi[k] = davidson(apply_H, psi[k];
                                  maxiter=20, tol=tol_davidson)

            # SVD verso sinistra
            rinds = (k == N) ? [siteind(psi, k)] :
                               [siteind(psi, k), linkind(psi, k)]
            U, S, V = svd(psi[k], rinds;
                          maxdim=maxdim, cutoff=cutoff,
                          righttags="Link,l=$(k-1)")
            psi[k] = V
            psi[k-1] = psi[k-1] * U * S

            # Aggiorna il blocco destro
            R[k] = (k == N) ?
                psi[k] * H[k] * dag(prime(psi[k])) :
                psi[k] * H[k] * dag(prime(psi[k])) * R[k+1]
        end

        @printf("Sweep %2d | E = %.12f | maxdim = %d\n",
                sweep, E, maxlinkdim(psi))
    end

    return E, psi
end

# ─────────────────────────────────────────────
# 6. Main: catena XX con N=20
# ─────────────────────────────────────────────

let
    N = 20
    sites = siteinds("S=1/2", N; conserve_qns=false)

    # Hamiltoniana XX
    H = xx_mpo(sites; J=1.0)

    # Stato iniziale: Néel |↑↓↑↓…⟩
    state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    psi0 = MPS(sites, state)

    # DMRG
    E, psi = dmrg_singlesite(H, psi0;
                              nsweeps=20,
                              maxdim=64,
                              cutoff=1e-10)

    println("\nEnergia ground state: ", E)
    println("Bond dimension massima: ", maxlinkdim(psi))

    # Confronto con la soluzione esatta (fermioni liberi)
    E_exact = sum(2*cos(π*k/(N+1)) for k in 1:N÷2)
    println("Energia esatta (N/2 modi):  ", E_exact)
    println("Errore:                     ", abs(E - E_exact))
end
