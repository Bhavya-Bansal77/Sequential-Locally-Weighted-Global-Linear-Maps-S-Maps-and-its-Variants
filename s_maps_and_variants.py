import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from scipy.spatial import cKDTree
import warnings

# ==========================================
# 1. Data Generation (Lorenz System + Noise)
# ==========================================
def lorenz(state, t, sigma=10.0, beta=8./3., rho=28.0):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

np.random.seed(42)
dt = 0.02
t = np.arange(0.0, 1000.0, dt)
trajectory = odeint(lorenz, [1.0, 1.0, 1.0], t)

# 5% noise to demonstrate the value of Regularization
noise_X = 0.05 * np.std(trajectory[:, 0])
noise_Y = 0.05 * np.std(trajectory[:, 1])
noise_Z = 0.05 * np.std(trajectory[:, 2])

X_noisy = trajectory[:, 0] + np.random.normal(0, noise_X, len(t))
Y_noisy = trajectory[:, 1] + np.random.normal(0, noise_Y, len(t))
Z_noisy = trajectory[:, 2] + np.random.normal(0, noise_Z, len(t))
X_true = trajectory[:, 0]

# ==========================================
# 2. Dynamic Inference: AMI & FNN
# ==========================================
def compute_ami(x, max_lag=50, bins=20):
    ami = np.zeros(max_lag)
    for lag in range(1, max_lag):
        c_xy, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=bins)
        p_xy = c_xy / np.sum(c_xy)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        mask = p_xy > 0
        ami[lag] = np.sum(p_xy[mask] * np.log2(p_xy[mask] / (p_x[:, None] * p_y[None, :])[mask]))
    
    minima = np.where((ami[1:-1] < ami[:-2]) & (ami[1:-1] < ami[2:]))[0] + 1
    return minima[0] if len(minima) > 0 else np.argmin(ami[1:]) + 1

def compute_fnn(x, tau, max_E=8, R_tol=15.0, A_tol=2.0):
    Ra = np.std(x)
    fnns = []
    for E in range(1, max_E + 1):
        N = len(x) - E * tau
        emb = np.column_stack([x[i*tau : N+i*tau] for i in range(E)])
        emb_plus = np.column_stack([x[i*tau : N+i*tau] for i in range(E+1)])
        
        tree = cKDTree(emb)
        dists, indices = tree.query(emb, k=2)
        nn_dists = dists[:, 1]
        nn_indices = indices[:, 1]
        
        d_plus = np.abs(emb_plus[:, -1] - emb_plus[nn_indices, -1])
        
        ratio = d_plus / np.maximum(nn_dists, 1e-10)
        cond1 = ratio > R_tol
        cond2 = (np.sqrt(nn_dists**2 + d_plus**2) / Ra) > A_tol
        
        fnn_frac = np.mean(cond1 | cond2)
        fnns.append(fnn_frac)
        
        if fnn_frac < 0.05:
            return E, fnns
            
    return max_E, fnns

tau = compute_ami(X_noisy, max_lag=50)
E, _ = compute_fnn(X_noisy, tau, max_E=8)
print(f"Dynamically Inferred Parameters -> Tau: {tau}, E: {E}")

# ==========================================
# 3. Dynamic State Space Setup
# ==========================================
tp = 1 # Prediction horizon
valid_idx = np.arange((E-1)*tau, len(t) - tp)

SS_uni = np.column_stack([X_noisy[valid_idx - i*tau] for i in range(E)])
SS_multi = np.column_stack([X_noisy[valid_idx], Y_noisy[valid_idx], Z_noisy[valid_idx]])

# Targets
Target_future = X_noisy[valid_idx + tp]         # 1D Target (for Univariate & Viz)
True_future = X_true[valid_idx + tp]

# 3D Targets for Multivariate Autonomous Forecasting
Target_multi_future = np.column_stack([
    X_noisy[valid_idx + tp], 
    Y_noisy[valid_idx + tp], 
    Z_noisy[valid_idx + tp]
])

# ==========================================
# 4. Autonomous S-Map Engines
# ==========================================

def forecast_smap_uni_autonomous(lib_states, lib_targets, initial_history, forecast_steps, tau, E, theta, regularized=False, alpha=10.0):
    """Iterates univariate forecast, updating the delay coordinates dynamically."""
    history = list(initial_history)
    preds = []
    
    for _ in range(forecast_steps):
        current_t = len(history) - 1
        target_state = np.array([history[current_t - i*tau] for i in range(E)]) # Target vector
        
        dists = np.linalg.norm(lib_states - target_state, axis=1) # Calculate distance of target vector from all the points in the library
        d_mean = np.mean(dists) if np.mean(dists) > 0 else 1.0 # Mean distance
        weights = np.exp(-theta * dists / d_mean) # Formulate weights based on distances
        
        W = weights[:, None]
        AW = lib_states * W
        bW = lib_targets * weights
        
        if regularized:
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(AW, bW)
            pred = clf.predict(target_state.reshape(1, -1))[0]
        else:
            AW_int = np.c_[np.ones(AW.shape[0]), AW]
            c, _, _, _ = np.linalg.lstsq(AW_int, bW, rcond=None)
            pred = np.dot(np.append(1, target_state), c)
            
        preds.append(pred)
        history.append(pred)
        
    return np.array(preds)

def forecast_smap_multi_autonomous(lib_states, lib_targets_3d, initial_state, forecast_steps, theta, regularized=False, alpha=10.0):
    """Iterates multivariate forecast by predicting [X, Y, Z] simultaneously to update the state."""
    current_state = np.array(initial_state)
    preds_x = []
    
    for _ in range(forecast_steps):
        dists = np.linalg.norm(lib_states - current_state, axis=1)
        d_mean = np.mean(dists) if np.mean(dists) > 0 else 1.0
        weights = np.exp(-theta * dists / d_mean)
        
        W = weights[:, None]
        AW = lib_states * W
        bW = lib_targets_3d * weights[:, None]
        
        if regularized:
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(AW, bW)
            pred_state = clf.predict(current_state.reshape(1, -1))[0]
        else:
            AW_int = np.c_[np.ones(AW.shape[0]), AW]
            c, _, _, _ = np.linalg.lstsq(AW_int, bW, rcond=None)
            pred_state = np.dot(np.append(1, current_state), c)
            
        preds_x.append(pred_state[0])
        current_state = pred_state
        
    return np.array(preds_x)


# Setup train/test splits
train_end = 10000
test_start = 10000
test_end = 11000 
forecast_steps = test_end - test_start

SS_uni_train = SS_uni[:train_end]
SS_multi_train = SS_multi[:train_end]

Target_future_train = Target_future[:train_end]
Target_multi_train = Target_multi_future[:train_end]

# Initial COnditions for Autonomous FOrecasting
idx_test_start = valid_idx[test_start]
initial_history_uni = X_noisy[:idx_test_start + 1].tolist()
initial_state_multi = SS_multi[test_start]


# ==========================================
# Visualization & Plotting 
# ==========================================
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 12, hspace=0.3, wspace=0.05)
plt.rcParams.update({'font.size': 10})

target_i = 1500
theta_vis = 4.0 # Hard coded theta but can be inferred using LOOCV
opt_elev = 18
opt_azim = -140

# ==========================================
# ROW 1: Univariate S-Map 
# ==========================================
ax1 = fig.add_subplot(gs[0, 0:2], projection='3d')
ax2 = fig.add_subplot(gs[0, 3:5], projection='3d')

target_pt_uni = SS_uni_train[target_i]
dists_uni = np.linalg.norm(SS_uni_train - target_pt_uni, axis=1)
weights_uni = np.exp(-theta_vis * dists_uni / np.mean(dists_uni))
mask_uni = weights_uni > 0.05

# --- Col 1: Phase Space ---
ax1.plot(SS_uni_train[:, 0], SS_uni_train[:, 1], SS_uni_train[:, 2], color='#cccccc', linewidth=0.8)
ax1.scatter(SS_uni_train[mask_uni, 0], SS_uni_train[mask_uni, 1], SS_uni_train[mask_uni, 2], 
            c=weights_uni[mask_uni], cmap='Blues', s=15, alpha=1.0)
ax1.scatter(*target_pt_uni[:3], color='cyan', marker='*', s=300, edgecolor='black', zorder=10)
ax1.set_title(f"1A. Univariate Phase Space\n(Calculated: E={E}, $\\tau$={tau})", fontweight='bold')
ax1.view_init(elev=opt_elev, azim=opt_azim)

zoom = 3 
ax1.set_xlim(target_pt_uni[0]-zoom, target_pt_uni[0] + 3*zoom)
ax1.set_zlim(target_pt_uni[2] - 2*zoom, target_pt_uni[2] + 2*zoom)
ax1.set_axis_off()

# --- Col 2: Tangent Plane ---
W_mat_uni = weights_uni[:, None]
AW_uni = SS_uni_train * W_mat_uni
bW_uni = Target_future_train * weights_uni
AW_int_uni = np.c_[np.ones(AW_uni.shape[0]), AW_uni]
c_uni, _, _, _ = np.linalg.lstsq(AW_int_uni, bW_uni, rcond=None)

jac_uni_str = "DF = [" + ", ".join([f"{c_uni[i]:.2f}" for i in range(1, len(c_uni))]) + "]"

ax2.plot(SS_uni_train[:, 0], SS_uni_train[:, 1], Target_future_train, color='#e0e0e0', linewidth=1.0)
ax2.scatter(target_pt_uni[0], target_pt_uni[1], Target_future_train[target_i], color='cyan', marker='*', s=500, edgecolor='black', zorder=50)

delta_uni = 6.0 
u_uni = np.linspace(target_pt_uni[0]-delta_uni, target_pt_uni[0]+delta_uni, 10)
v_uni = np.linspace(target_pt_uni[1]-delta_uni, target_pt_uni[1]+delta_uni, 10)
U_uni, V_uni = np.meshgrid(u_uni, v_uni)

W_plane_uni = c_uni[0] + c_uni[1]*U_uni + c_uni[2]*V_uni
for i in range(3, E + 1):
    W_plane_uni += c_uni[i] * target_pt_uni[i-1] 

ax2.plot_wireframe(U_uni, V_uni, W_plane_uni, color='#1f77b4', linewidth=1.2, rstride=1, cstride=1)

ax2.set_xlim([target_pt_uni[0]-10, target_pt_uni[0]+7])
ax2.set_ylim([target_pt_uni[1]-7, target_pt_uni[1]+7])
ax2.set_zlim([Target_future_train[target_i]-7, Target_future_train[target_i]+7])
ax2.set_title(f"1B. Univariate Local Jacobian\n{jac_uni_str}", fontweight='bold')
ax2.set_xlabel(r"$X(t)$")
ax2.set_ylabel(r"$X(t-\tau)$")
ax2.set_zlabel(r"$X(t+tp)$")
ax2.view_init(elev=15, azim=110)
ax2.set_axis_on()


time_window = np.arange(test_end - test_start)


# ==========================================
# ROW 2: Multivariate S-Map 
# ==========================================
ax4 = fig.add_subplot(gs[1, 0:2], projection='3d')
ax5 = fig.add_subplot(gs[1, 3:5], projection='3d')

target_pt_multi = SS_multi_train[target_i]
dists_multi = np.linalg.norm(SS_multi_train - target_pt_multi, axis=1)
weights_multi = np.exp(-theta_vis * dists_multi / np.mean(dists_multi))
mask_multi = weights_multi > 0.05

# --- Col 1: Phase Space ---
ax4.plot(SS_multi_train[:, 0], SS_multi_train[:, 1], SS_multi_train[:, 2], color='#cccccc', linewidth=0.8)
ax4.scatter(SS_multi_train[mask_multi, 0], SS_multi_train[mask_multi, 1], SS_multi_train[mask_multi, 2], 
            c=weights_multi[mask_multi], cmap='Greens', s=15, alpha=1.)
ax4.scatter(*target_pt_multi, color='red', marker='*', s=300, edgecolor='black', zorder=10)
ax4.set_title("2A. Multivariate Phase Space\n(X, Y, Z)", fontweight='bold')
ax4.view_init(elev=opt_elev, azim=-45)

zoom_multi = 6  
ax4.set_xlim(target_pt_multi[0] - 2, target_pt_multi[0] + 2*zoom_multi)
ax4.set_ylim(target_pt_multi[1] - zoom_multi, target_pt_multi[1] + zoom_multi)
ax4.set_zlim(target_pt_multi[2] - zoom_multi, target_pt_multi[2] + zoom_multi)
ax4.set_axis_off()

# --- Col 2: Tangent Plane ---
W_mat_multi = weights_multi[:, None]
AW_multi = SS_multi_train * W_mat_multi
bW_multi = Target_future_train * weights_multi 
AW_int_multi = np.c_[np.ones(AW_multi.shape[0]), AW_multi]
c_multi, _, _, _ = np.linalg.lstsq(AW_int_multi, bW_multi, rcond=None)

jac_multi_str = f"DF = [{c_multi[1]:.2f}, {c_multi[2]:.2f}, {c_multi[3]:.2f}]"

ax5.plot(SS_multi_train[:, 1], SS_multi_train[:, 2], Target_future_train, color='#e0e0e0', linewidth=1.0)
ax5.scatter(target_pt_multi[1], target_pt_multi[2], Target_future_train[target_i], color='red', marker='*', s=500, edgecolor='black', zorder=50)

u = np.linspace(target_pt_multi[1]-(delta_uni + 5), target_pt_multi[1]+(delta_uni + 5), 10) 
v = np.linspace(target_pt_multi[2]-(delta_uni + 5), target_pt_multi[2]+(delta_uni + 5), 10) 
U, V = np.meshgrid(u, v)

W_plane = c_multi[0] + c_multi[1]*target_pt_multi[0] + c_multi[2]*U + c_multi[3]*V

ax5.plot_wireframe(U, V, W_plane, color='#2ca02c', linewidth=1.2, rstride=1, cstride=1)

ax5.set_xlim([target_pt_multi[1]-10, target_pt_multi[1]+10])
ax5.set_ylim([target_pt_multi[2]-20, target_pt_multi[2]+20])
ax5.set_zlim([Target_future_train[target_i]-10, Target_future_train[target_i]+10])
ax5.set_title(f"2B. Multivariate Local Jacobian\n{jac_multi_str}", fontweight='bold')
ax5.set_xlabel(r"$Y(t)$")
ax5.set_ylabel(r"$Z(t)$")
ax5.set_zlabel(r"$X(t+tp)$")
ax5.view_init(elev=15, azim=110)
ax5.set_axis_on()


# ==========================================
# ROW 3: Regularization on Autonomous Horizons
# ==========================================
ax7 = fig.add_subplot(gs[0, 6:12])
ax8 = fig.add_subplot(gs[1, 6:12])

true_process = True_future[test_start:test_end]

def safe_metrics(true_y, pred_y):
    """Safely calculates RMSE and Correlation, handling NaNs and extreme divergence."""
    valid = np.isfinite(pred_y) & np.isfinite(true_y)
    if np.sum(valid) < 2:
        return np.nan, np.nan
    rmse = np.sqrt(np.mean((true_process[valid] - pred_y[valid])**2))
    corr = np.corrcoef(true_process[valid], pred_y[valid])[0, 1]
    return rmse, corr

# --- Col 1: Univariate ---
std_uni_overfit = forecast_smap_uni_autonomous(
    SS_uni_train, Target_future_train, initial_history_uni, 
    forecast_steps, tau, E, theta=theta_vis, regularized=False
)
reg_uni = forecast_smap_uni_autonomous(
    SS_uni_train, Target_future_train, initial_history_uni, 
    forecast_steps, tau, E, theta=theta_vis, regularized=True, alpha=15.0
)

rmse_std_uni, corr_std_uni = safe_metrics(true_process, std_uni_overfit)
rmse_reg_uni, corr_reg_uni = safe_metrics(true_process, reg_uni)

ax7.plot(time_window, true_process, color='lightgray', lw=4.5, label="True Process")
ax7.plot(time_window, std_uni_overfit, color='#d62728', alpha=0.8, lw=1.5, 
         label=f"Standard S-Map (RMSE: {rmse_std_uni:.2f}, ρ: {corr_std_uni:.2f})")
ax7.plot(time_window, reg_uni, color='#1f77b4', lw=2.5, 
         label=f"Regularized S-Map (RMSE: {rmse_reg_uni:.2f}, ρ: {corr_reg_uni:.2f})")
ax7.axvline(x=80,ymin=0, ymax= 1.0/3.0, color='red', linestyle='--', label='Vertical Line')
ax7.scatter(x=80, y=-22, marker='*', color='red', clip_on=False, s=200, zorder=20)

ax7.set_ylim(np.min(true_process) - 5, np.max(true_process) + 5)
ax7.set_title("3A. Univariate Autonomous: Standard vs. Regularized", fontweight='bold')
ax7.legend(loc='best', fontsize=5)

# --- Col 2: Multivariate ---
std_multi_overfit = forecast_smap_multi_autonomous(
    SS_multi_train, Target_multi_train, initial_state_multi, 
    forecast_steps, theta=theta_vis, regularized=False
)
reg_multi = forecast_smap_multi_autonomous(
    SS_multi_train, Target_multi_train, initial_state_multi, 
    forecast_steps, theta=theta_vis, regularized=True, alpha=15.0
)

rmse_std_multi, corr_std_multi = safe_metrics(true_process, std_multi_overfit)
rmse_reg_multi, corr_reg_multi = safe_metrics(true_process, reg_multi)

ax8.plot(time_window, true_process, color='lightgray', lw=4.5, label="True Process")
ax8.plot(time_window, std_multi_overfit, color='#d62728', alpha=0.8, lw=1.5, 
         label=f"Standard S-Map (RMSE: {rmse_std_multi:.2f}, ρ: {corr_std_multi:.2f})")
ax8.plot(time_window, reg_multi, color='#2ca02c', lw=2.5, 
         label=f"Regularized S-Map (RMSE: {rmse_reg_multi:.2f}, ρ: {corr_reg_multi:.2f})")
ax8.axvline(x=100,ymin=0, ymax= 1.0/2.0, color='red', linestyle='--', label='Vertical Line')
ax8.scatter(x=100, y=-22, marker='*', color='red', clip_on=False, s=200, zorder=20)

ax8.set_ylim(np.min(true_process) - 5, np.max(true_process) + 5)
ax8.set_title("3B. Multivariate Autonomous: Standard vs. Regularized", fontweight='bold')
ax8.legend(loc='best', fontsize=5)
plt.tight_layout()
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.1)
plt.show()
