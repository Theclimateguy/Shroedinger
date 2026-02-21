# -------------------------
# Experiment E: coherence-driven vertical rates
# -------------------------
def exp_coherence_driven_rates(outdir: Path, seed: int = BASE_SEED):
    """
    Проверяет, насколько хорошо локальная когерентность и Λ-подобные скаляры
    предсказывают величину вертикального источника |Tr(h_b D_mu[ρ])|.
    
    Результаты сохраняются в CSV, включая корреляции и R^2 для разных предикторов.
    """
    rng = np.random.default_rng(seed)
    
    # Параметры системы (аналогично exp_full_balance_coherence_driven)
    L = 4
    K = 4
    xs = np.arange(L)
    ks = np.arange(K)
    
    def axis_mu(x, k):
        v = [
            np.sin(0.7*x + 0.9*k) + 0.25*np.cos(0.3*x - 0.4*k),
            np.cos(0.6*x - 0.8*k) + 0.15*np.sin(0.9*x + 0.2*k),
            np.sin(0.5*x + 0.5*k) + 0.35,
        ]
        return normalize_vec(v)
    
    def axis_x(x, k):
        v = [
            np.cos(0.8*x + 0.4*k) + 0.1,
            np.sin(0.5*x - 0.7*k) + 0.15*np.cos(0.9*x + 0.3*k),
            np.cos(0.4*x + 0.6*k) - 0.25,
        ]
        return normalize_vec(v)
    
    omega = 0.25 + 0.45*np.exp(-0.5*((ks-(K-1)/2)/(0.35*K))**2)
    angle_mu = 0.9 * omega/omega.max()
    angle_x = 0.55 * (0.75 + 0.25*np.cos(2*np.pi*ks/K))
    
    Ux = {}
    Umu = {}
    for x in xs:
        for k in ks:
            Umu[(x,k)] = su2_from_axis_angle(axis_mu(x,k), float(angle_mu[k]))
            Ux[(x,k)] = su2_from_axis_angle(axis_x(x,k), float(angle_x[k]))
    
    def P_xmu(x, k):
        xp = (x+1) % L
        kp = (k+1) % K
        return Ux[(x,k)] @ Umu[(xp,k)] @ Ux[(x,kp)].conj().T @ Umu[(x,k)].conj().T
    
    Fsite = {(x,k): F_phys_from_P(P_xmu(x,k)) for x in xs for k in ks}
    F2site = {(x,k): Fsite[(x,k)]@Fsite[(x,k)] for x in xs for k in ks}
    
    # Построение гамильтониана цепочки и токов
    d = 2**L
    D = K*d
    J = 1.0
    h_bonds = [bond_xx_yy(i, i+1, L, J) for i in range(L-1)]
    H_chain = sum(h_bonds)
    j_bonds = [1j * comm(h_bonds[i], h_bonds[i+1]) for i in range(L-2)]
    
    Pk = [np.eye(K, dtype=complex)[[k]].T @ np.eye(K, dtype=complex)[[k]] for k in range(K)]
    def lift_to_layer(op_chain, k):
        return np.kron(Pk[k], op_chain)
    
    H_tot = sum(lift_to_layer(H_chain, k) for k in range(K))
    gamma = 0.35
    Ls_deph = [np.sqrt(gamma)*lift_to_layer(op_on_site(SZ, site, L), k) for k in range(K) for site in range(L)]
    
    U_chain = {k: kronN([Umu[(x,k)] for x in xs]) for k in range(K-1)}
    
    def random_pure_qubit_local(rr):
        v = rr.normal(size=2) + 1j*rr.normal(size=2)
        v = v/np.linalg.norm(v)
        return v.reshape(2,1)@v.conj().reshape(1,2)
    
    def random_product_state_local(rr):
        return kronN([random_pure_qubit_local(rr) for _ in range(L)])
    
    Ns = 500   # больше выборок для статистики
    eta0_list = [0.5, 0.9, 1.5]  # разные базовые скорости
    
    rows = []
    for eta0 in eta0_list:
        for s in range(Ns):
            rr = np.random.default_rng(10000 + s + int(eta0*1000))
            w = rr.dirichlet(alpha=np.ones(K))
            rho_layers = [random_product_state_local(rr) for _ in range(K)]
            rho_tot = np.zeros((D,D), dtype=complex)
            for k in range(K):
                rho_tot[k*d:(k+1)*d, k*d:(k+1)*d] = w[k]*rho_layers[k]
            rho_tot = normalize_rho(rho_tot)
            
            # Когерентность на каждом слое
            coh_layer = np.zeros(K)
            for k in range(K):
                rho_k = rho_tot[k*d:(k+1)*d, k*d:(k+1)*d]
                pk = float(np.real(np.trace(rho_k)))
                if pk < 1e-14:
                    continue
                rho_kc = rho_k/pk
                cohs = [coherence_offdiag(partial_trace_site(rho_kc, x, L)) for x in xs]
                coh_layer[k] = float(np.mean(cohs))
            
            eta = [eta0 * min(1.0, coh_layer[k]/0.5) for k in range(K-1)]
            Ls_mu = []
            for k in range(K-1):
                shift = np.zeros((K,K), dtype=complex)
                shift[k+1, k] = 1.0
                Ls_mu.append(np.sqrt(eta[k]) * np.kron(shift, U_chain[k]))
            
            unitary_part = -1j*comm(H_tot, rho_tot)
            D_deph = lindblad_dissipator(rho_tot, Ls_deph)
            D_mu = lindblad_dissipator(rho_tot, Ls_mu)
            dr = unitary_part + D_deph + D_mu
            
            for k in range(K):
                rho_k = rho_tot[k*d:(k+1)*d, k*d:(k+1)*d]
                pk_layer = float(np.real(np.trace(rho_k)))
                if pk_layer < 1e-14:
                    continue
                rho_kc = rho_k/pk_layer
                site_red = {x: partial_trace_site(rho_kc, x, L) for x in xs}
                site_coh = {x: coherence_offdiag(site_red[x]) for x in xs}
                site_L = {x: float(np.real(np.trace(Fsite[(x,k)] @ site_red[x]))) for x in xs}
                
                for b in range(L-1):
                    O = lift_to_layer(h_bonds[b], k)
                    src_mu = expect(O, D_mu)
                    # признаки
                    coh_mean_layer = coh_layer[k]
                    coh_sum = site_coh[b] + site_coh[b+1]
                    absLambda_sum = abs(site_L[b]) + abs(site_L[b+1])
                    
                    rows.append({
                        'eta0': eta0,
                        'sample': s,
                        'k': k,
                        'bond': b,
                        'abs_src_mu': abs(src_mu),
                        'coh_layer_mean': coh_mean_layer,
                        'coh_sum': coh_sum,
                        'absLambda_sum': absLambda_sum,
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'coherence_driven_rates_raw.csv', index=False)
    
    # Вычисление корреляций и R^2 для каждого eta0
    summaries = []
    for eta0 in eta0_list:
        df_sub = df[df['eta0'] == eta0]
        y = df_sub['abs_src_mu'].values
        X_coh = df_sub[['coh_layer_mean', 'coh_sum']].values
        X_lam = df_sub[['absLambda_sum']].values
        
        # Корреляции
        corr_coh = np.corrcoef(df_sub['coh_sum'].values, y)[0,1]
        corr_lam = np.corrcoef(df_sub['absLambda_sum'].values, y)[0,1]
        
        # Простая линейная регрессия для одного предиктора (можно расширить)
        from sklearn.linear_model import LinearRegression
        reg_coh = LinearRegression().fit(X_coh, y)
        r2_coh = reg_coh.score(X_coh, y)
        reg_lam = LinearRegression().fit(X_lam, y)
        r2_lam = reg_lam.score(X_lam, y)
        
        summaries.append({
            'eta0': eta0,
            'corr_coh_sum': corr_coh,
            'corr_absLambda_sum': corr_lam,
            'R2_coh_features': r2_coh,
            'R2_absLambda_only': r2_lam,
        })
    
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(outdir / 'coherence_driven_rates_summary.csv', index=False)
    
    print(f"Experiment E: data saved to {outdir}/coherence_driven_rates_*.csv")