‎
‎# FILE: diff_ccz4_gradient_descent_test_v32_AggressiveLR.py
‎
‎import jax
‎import jax.numpy as jnp
‎from jax import jit, grad, checkpoint
‎from jax.lax import fori_loop, cond
‎from jax.tree_util import register_pytree_node_class, tree_map 
‎from functools import partial
‎import time
‎
‎# =============================================================================
‎# --- 0. KONFIGURASI DAN PARAMETER ---
‎# =============================================================================
‎N = 10          
‎dx = 0.1
‎FIELD_SHAPE = (N, N, N)
‎DIM = 3
‎
‎@register_pytree_node_class
‎class Params:
‎    def __init__(self, kappa1, kappa2, eta):
‎        self.kappa1 = jnp.asarray(kappa1, dtype=jnp.float32)
‎        self.kappa2 = jnp.asarray(kappa2, dtype=jnp.float32)
‎        self.eta = jnp.asarray(eta, dtype=jnp.float32)
‎        
‎    def replace(self, **kwargs):
‎        return Params(
‎            kwargs.get('kappa1', self.kappa1),
‎            kwargs.get('kappa2', self.kappa2),
‎            kwargs.get('eta', self.eta)
‎        )
‎
‎    def tree_flatten(self):
‎        return (self.kappa1, self.kappa2, self.eta), None
‎    
‎    @classmethod
‎    def tree_unflatten(cls, aux_data, children):
‎        return cls(*children)
‎
‎GLOBAL_DT = 0.005
‎
‎PARAMS = Params(kappa1=2.0, kappa2=1.0, eta=2.0)
‎
‎TEST_TIMESTEPS = 200      
‎CHECKPOINT_INTERVAL = 5   
‎# KOREKSI KRITIS: Tingkatkan Learning Rate secara drastis untuk mengatasi gradien 1e-18
‎LEARNING_RATE = 1.0       
‎N_ITERATIONS = 200        
‎
‎# =============================================================================
‎# --- OPTIMIZER ADAM DARI JAX/OPTICS ---
‎# =============================================================================
‎@register_pytree_node_class
‎class AdamState:
‎    def __init__(self, count, m_eta, v_eta):
‎        self.count = count
‎        self.m_eta = m_eta
‎        self.v_eta = v_eta
‎        
‎    def replace(self, **kwargs):
‎            return AdamState(
‎                kwargs.get('count', self.count),
‎                kwargs.get('m_eta', self.m_eta),
‎                kwargs.get('v_eta', self.v_eta)
‎            )
‎
‎    def tree_flatten(self):
‎        children = (self.count, self.m_eta, self.v_eta)
‎        return children, None
‎    
‎    @classmethod
‎    def tree_unflatten(cls, aux_data, children):
‎        return cls(*children)
‎
‎def adam_init(params):
‎    return AdamState(jnp.array(0), jnp.zeros_like(params.eta, dtype=jnp.float32), jnp.zeros_like(params.eta, dtype=jnp.float32))
‎
‎def adam_update(i, grad_eta, state, learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8):
‎    # Menggunakan learning_rate yang disupply oleh argumen fungsi luar
‎    lr_val = learning_rate 
‎    
‎    state = state.replace(count=state.count + 1)
‎    i = state.count
‎    
‎    m = (b1 * state.m_eta) + (1 - b1) * grad_eta
‎    v = (b2 * state.v_eta) + (1 - b2) * (grad_eta * grad_eta)
‎    
‎    m_hat = m / (1 - b1**i)
‎    v_hat = v / (1 - b2**i)
‎    
‎    # Menggunakan lr_val yang disupply
‎    delta_eta = lr_val * m_hat / (jnp.sqrt(v_hat) + eps)
‎    
‎    new_state = AdamState(state.count, m, v)
‎    return new_state, delta_eta 
‎
‎# =============================================================================
‎# --- 1. CCZ4 STATE (PYTREE) & INITIALIZATION ---
‎# =============================================================================
‎@register_pytree_node_class
‎class CCZ4State:
‎    def __init__(self, phi, chi, K, At, Gam, Theta, Z, alpha, beta):
‎        self.phi, self.chi, self.K, self.At, self.Gam, self.Theta, self.Z, self.alpha, self.beta = \
‎            phi, chi, K, At, Gam, Theta, Z, alpha, beta
‎    def to_tuple(self):
‎        return (self.phi, self.chi, self.K, self.At, self.Gam, self.Theta, self.Z, self.alpha, self.beta)
‎    @classmethod
‎    def from_tuple(cls, t):
‎        return cls(*t)
‎    def tree_flatten(self):
‎        return self.to_tuple(), None
‎    @classmethod
‎    def tree_unflatten(cls, aux_data, children):
‎        return cls(*children)
‎
‎def initialize_minkowski(N):
‎    zeros = jnp.zeros(FIELD_SHAPE, dtype=jnp.float32)
‎    ones = jnp.ones(FIELD_SHAPE, dtype=jnp.float32)
‎    r = jnp.linalg.norm(jnp.mgrid[:N,:N,:N] - N/2, axis=0) * dx
‎    r = jnp.asarray(r, dtype=jnp.float32)
‎    alpha_init = ones - 0.01 * jnp.exp(-r**2 / 5.0)
‎    perturbation = 1e-2 * jnp.exp(-r**2 / 5.0)
‎    K_init = zeros + perturbation
‎    return CCZ4State(zeros, ones, K_init, jnp.zeros((5,) + FIELD_SHAPE, dtype=jnp.float32),
‎                     jnp.zeros((3,) + FIELD_SHAPE, dtype=jnp.float32), zeros, jnp.zeros((3,) + FIELD_SHAPE, dtype=jnp.float32),
‎                     alpha_init, jnp.zeros((3,) + FIELD_SHAPE, dtype=jnp.float32))
‎
‎# =============================================================================
‎# --- 2. ALGEBRA TENSOR RIIL & CONSTRAINTS ---
‎# =============================================================================
‎def diff6(f, dx):
‎    d_x = jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)
‎    d_y = jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)
‎    d_z = jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)
‎    return jnp.stack([d_x, d_y, d_z], axis=0) / (2.0 * dx)
‎def grad6(f): return diff6(f, dx)
‎def advect(f, beta, d):
‎    grad_f = d(f)
‎    return beta[0]*grad_f[0] + beta[1]*grad_f[1] + beta[2]*grad_f[2]
‎
‎@jit
‎def voigt_to_tensor(A_voigt):
‎    shape = A_voigt.shape[1:]
‎    A_full = jnp.zeros((DIM, DIM) + shape, dtype=jnp.float32)
‎    A_full = A_full.at[0,0].set(A_voigt[0])
‎    A_full = A_full.at[1,1].set(A_voigt[3])
‎    A_full = A_full.at[2,2].set(-A_voigt[0] - A_voigt[3])
‎    A_full = A_full.at[0,1].set(A_voigt[1]); A_full = A_full.at[1,0].set(A_voigt[1])
‎    A_full = A_full.at[0,2].set(A_voigt[2]); A_full = A_full.at[2,0].set(A_voigt[2])
‎    A_full = A_full.at[1,2].set(A_voigt[4]); A_full = A_full.at[2,1].set(A_voigt[4])
‎    return A_full
‎
‎@jit
‎def get_gamma_inv(state):
‎    return jnp.identity(DIM, dtype=jnp.float32)[:, :, None, None, None] * jnp.ones((DIM, DIM) + FIELD_SHAPE, dtype=jnp.float32)
‎
‎@jit
‎def hamiltonian_constraint(state):
‎    chi = state.chi
‎    At_full = voigt_to_tensor(state.At)
‎    gamma_inv = get_gamma_inv(state)
‎    At_sq = jnp.einsum('ij..., kl..., ik..., jl... -> ...',
‎                       At_full, At_full, gamma_inv, gamma_inv)
‎    log_chi = jnp.log(chi + 1e-15)
‎    d_log_chi = grad6(log_chi)
‎    grad_sq_sum = jnp.sum(d_log_chi * d_log_chi, axis=0)
‎    R_chi = -8 * grad_sq_sum 
‎    H = (R_chi + state.K**2 - At_sq) / chi + 2.0 * state.K * state.Theta 
‎    return H
‎
‎@jit
‎def momentum_constraint(state):
‎    At_full = voigt_to_tensor(state.At)
‎    d_At = grad6(At_full)
‎    Mi = jnp.einsum('jk..., kij... -> i...',
‎                    get_gamma_inv(state), d_At)
‎    return Mi * 0.1
‎
‎@jit
‎def tensor_to_voigt_rhs(A_full):
‎    """Mengambil tensor (3, 3, N, N, N) dan mengembalikan 5 komponen Voigt (5, N, N, N)."""
‎    return jnp.stack([
‎        A_full[0,0],
‎        A_full[0,1],
‎        A_full[0,2],
‎        A_full[1,1],
‎        A_full[1,2]
‎    ], axis=0)
‎
‎# =============================================================================
‎# --- 3. CCZ4 RHS (MASTER EVOLUTION EQUATIONS) ---
‎# =============================================================================
‎def ccz4_rhs(state_tuple, dx, params):
‎    state = CCZ4State.from_tuple(state_tuple)
‎    d = lambda f: diff6(f, dx)
‎    Lie = lambda f: advect(f, state.beta, d)
‎    H = hamiltonian_constraint(state)
‎    Mi = momentum_constraint(state)
‎    
‎    alpha = state.alpha
‎    K = state.K
‎    At_full = voigt_to_tensor(state.At)
‎    
‎    dTheta_dt = (
‎        Lie(state.Theta)
‎        + 0.5 * alpha * H
‎        - alpha * (2.0 - params.kappa1) * K * state.Theta
‎        - params.kappa1 * alpha * state.Theta
‎    )
‎    dZ_dt = (
‎        Lie(state.Z)
‎        + alpha * (Mi - (2/3) * d(K) * state.Theta)
‎        - params.kappa2 * alpha * state.Z 
‎    )
‎
‎    dphi_dt = jnp.zeros_like(state.phi)
‎    dchi_dt = Lie(state.chi) + (1.0 / 3.0) * state.chi * alpha * K
‎    dK_dt = (
‎        Lie(K)
‎        + alpha * (K**2 / 3.0 + H / 2.0)
‎        + alpha * 0.0
‎    )
‎
‎    Ricc_proxy = K * K * jnp.identity(DIM, dtype=jnp.float32)[:, :, None, None, None] * 0.1
‎    dAt_dt = (
‎        Lie(At_full)
‎        - alpha * (Ricc_proxy - K * At_full)
‎        - 2.0 * alpha * K * At_full 
‎    )
‎    
‎    dAt_dt_voigt = tensor_to_voigt_rhs(dAt_dt)
‎
‎    dGam_dt = Lie(state.Gam) - params.eta * alpha * state.Gam + alpha * Mi 
‎    dalpha_dt = -2.0 * alpha * K
‎    dbeta_dt = 0.75 * state.Gam - params.eta * state.beta 
‎
‎    return (dphi_dt, dchi_dt, dK_dt, dAt_dt_voigt, dGam_dt, dTheta_dt, dZ_dt, dalpha_dt, dbeta_dt)
‎
‎
‎# =============================================================================
‎# --- 4. INTEGRATOR & STEP FUNCTIONS ---
‎# =============================================================================
‎def rk4_step(state_tuple, rhs_fn, dt, dx, params):
‎    
‎    def add_scaled(state, k, scale):
‎        return tree_map(lambda x, y: x + scale * y, state, k)
‎
‎    dt_f = jnp.float32(dt)
‎    
‎    k1 = rhs_fn(state_tuple, dx, params)
‎    k2 = rhs_fn(add_scaled(state_tuple, k1, dt_f * 0.5), dx, params)
‎    k3 = rhs_fn(add_scaled(state_tuple, k2, dt_f * 0.5), dx, params)
‎    k4 = rhs_fn(add_scaled(state_tuple, k3, dt_f), dx, params)
‎    
‎    final_step = tree_map(lambda k1_c, k2_c, k3_c, k4_c: (dt_f / 6.0) * (k1_c + 2*k2_c + 2*k3_c + k4_c),
‎                          k1, k2, k3, k4)
‎    
‎    return tree_map(lambda s, f: s + f, state_tuple, final_step)
‎
‎def ccz4_step(state_tuple, dt, dx, params):
‎    return rk4_step(state_tuple, ccz4_rhs, dt, dx, params)
‎
‎checkpointed_step = checkpoint(ccz4_step)
‎
‎# =============================================================================
‎# --- 5. LOOPING (EVOLUTION) ---
‎# =============================================================================
‎def raw_full_adjoint_evolution(initial_state_tuple, num_steps, dt, dx, params, checkpoint_interval):
‎    
‎    initial_loop_state = (initial_state_tuple, dt, params)
‎    
‎    def single_step_pytree(loop_state):
‎        state, dt_val, params_val = loop_state
‎        new_state = ccz4_step(state, dt_val, dx, params_val)
‎        return (new_state, dt_val, params_val)
‎
‎    def checkpointed_step_pytree(loop_state):
‎        state, dt_val, params_val = loop_state
‎        new_state = checkpointed_step(state, dt_val, dx, params_val)
‎        return (new_state, dt_val, params_val)
‎    
‎    def body(i, loop_state):
‎        return cond(i % checkpoint_interval == 0,
‎                    checkpointed_step_pytree,
‎                    single_step_pytree,       
‎                    loop_state)
‎    
‎    final_loop_state = fori_loop(0, num_steps, body, initial_loop_state)
‎    final_state_tuple, _, _ = final_loop_state
‎    return final_state_tuple
‎
‎# Currying dan JIT untuk mengisolasi argumen statis loop
‎full_adjoint_evolution = partial(jit, static_argnames=('num_steps', 'dx', 'checkpoint_interval'))(raw_full_adjoint_evolution)
‎
‎# =============================================================================
‎# --- 6. LOSS FUNCTION DAN GRADIENT ---
‎# =============================================================================
‎def raw_loss_fn(initial_state_tuple, params, TEST_TIMESTEPS, GLOBAL_DT, dx, CHECKPOINT_INTERVAL):
‎    """Loss: meminimalkan pelanggaran Constraint dan K + Momentum Constraint."""
‎    
‎    final = full_adjoint_evolution(initial_state_tuple, TEST_TIMESTEPS, GLOBAL_DT,
‎                                   dx, params, CHECKPOINT_INTERVAL)
‎    
‎    state = CCZ4State.from_tuple(final)
‎    Mi_sq = jnp.mean(momentum_constraint(state)**2)
‎    
‎    # Menggunakan bobot K^2 = 1.0 dari langkah sebelumnya
‎    return jnp.mean(state.Theta**2) + 1.0 * jnp.mean(state.K**2) + Mi_sq
‎
‎# 1. Loss Function Utama (di-curry dan JIT)
‎jitted_loss_fn = jit(partial(raw_loss_fn, TEST_TIMESTEPS=TEST_TIMESTEPS, GLOBAL_DT=GLOBAL_DT,
‎                                          dx=dx, CHECKPOINT_INTERVAL=CHECKPOINT_INTERVAL))
‎
‎# 2. Gradien terhadap PyTree params (indeks 1)
‎grad_params_raw = grad(lambda state, p: raw_loss_fn(state, p, TEST_TIMESTEPS, GLOBAL_DT, dx, CHECKPOINT_INTERVAL), argnums=1)
‎
‎# 3. Wrapper untuk mendapatkan gradien eta
‎def get_eta_gradient(initial_state_tuple, params):
‎    grad_ptree = grad_params_raw(initial_state_tuple, params)
‎    return grad_ptree.eta
‎
‎# =============================================================================
‎# --- 7. GRADIENT DESCENT BENCHMARK ---
‎# =============================================================================
‎
‎@jit
‎def optimization_step(current_params, current_state_tuple):
‎    
‎    grad_eta_raw = get_eta_gradient(current_state_tuple, current_params)
‎    grad_eta = jnp.float32(grad_eta_raw)
‎    
‎    loss_val = jitted_loss_fn(current_state_tuple, current_params)
‎    return loss_val, grad_eta 
‎
‎def run_gradient_descent_benchmark():
‎    global N, dx, FIELD_SHAPE, CHECKPOINT_INTERVAL, TEST_TIMESTEPS, LEARNING_RATE, N_ITERATIONS
‎    N_start = N
‎    
‎    FIELD_SHAPE = (N, N, N)
‎
‎    initial_state = initialize_minkowski(N)
‎    state_tuple = initial_state.to_tuple()
‎    
‎    params_opt = PARAMS 
‎    opt_state = adam_init(params_opt)
‎    
‎    print(f"--- 🏆 DIFF-CCZ4 GRADIENT DESCENT TEST N={N_start}^3 / T={TEST_TIMESTEPS} (K1: {PARAMS.kappa1.item():.1f}, K2: {PARAMS.kappa2.item():.1f}, ETA_INIT: {PARAMS.eta.item():.1f}, LR: {LEARNING_RATE:.0e}, ITER: {N_ITERATIONS}) ---")
‎    
‎    # --- Kompilasi Awal ---
‎    print("Compiling loss function...", end="")
‎    try:
‎        loss_val_dummy, grad_eta_dummy = optimization_step(params_opt, state_tuple)
‎        loss_val_dummy.block_until_ready()
‎        print("DONE")
‎    except Exception as e:
‎        print(f"ERROR: Kompilasi Gagal: {e}")
‎        return
‎
‎    loss_history = []
‎    
‎    # MAIN OPTIMIZATION LOOP
‎    for step in range(1, N_ITERATIONS + 1):
‎        start_step = time.time()
‎        
‎        loss_val, grad_eta = optimization_step(params_opt, state_tuple)
‎        
‎        # Menggunakan LEARNING_RATE = 1.0
‎        opt_state, delta_eta = adam_update(step, grad_eta, opt_state, learning_rate=LEARNING_RATE)
‎        
‎        new_eta = params_opt.eta - delta_eta
‎        params_opt = params_opt.replace(eta=new_eta)
‎        
‎        loss_val.block_until_ready()
‎        step_time = time.time() - start_step
‎        loss_history.append(loss_val)
‎        
‎        if step % 50 == 0 or step <= 10:
‎            print(f"Step {step:03d} | Loss: {loss_val.item():.2e} | Grad E: {grad_eta.item():.2e} | Time: {step_time:.2f}s | E: {params_opt.eta.item():.4f}")
‎        
‎    print("\n--- ✅ HASIL AKHIR OPTIMASI ---")
‎    print(f"Loss Awal (Step 1): {loss_history[0].item():.2e}")
‎    print(f"Loss Akhir (Step {N_ITERATIONS}): {loss_history[-1].item():.2e}")
‎    print(f"Nilai Akhir Eta: {params_opt.eta.item():.4f}")
‎    
‎    if loss_history[-1] < loss_history[0] * 0.5:
‎        print("STATUS: SUCCESS. Loss turun signifikan. E bergerak ke nilai optimal.")
‎    elif loss_history[-1] < loss_history[0]:
‎        print("STATUS: SUCCESS. Loss turun. E bergerak ke nilai yang benar.")
‎    else:
‎        print("STATUS: CAUTION. Periksa apakah Loss atau E bergerak ke arah yang salah.")
‎
‎if __name__ == "__main__":
‎    run_gradient_descent_benchmark()
‎
