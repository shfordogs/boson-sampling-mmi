import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import json, time
import shutil, os, time
import ipynbname

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def u2mzi(dim, m, n, theta, phi, lp=torch.tensor(1, dtype=torch.complex128), lc=torch.tensor(1, dtype=torch.complex128)):
    assert m < n < dim
    mat = torch.eye(dim, dtype=torch.complex128).to(theta.device)
    sqrt_lp = torch.sqrt(lp).to(phi.device)
    sqrt_lc = torch.sqrt(lc).to(phi.device)
    phase = torch.exp(1j * theta)

    mat[m, m] = sqrt_lp * phase * torch.sin(phi)
    mat[m, n] = sqrt_lc * phase * torch.cos(phi)
    mat[n, m] = sqrt_lc * torch.cos(phi)
    mat[n, n] = -sqrt_lp * torch.sin(phi)
    return mat

def angle_diff(comp_src, comp_dst, offset=0, tolerance=torch.tensor(1e-6), wrap=True, to_degree=False):
    zero_src = torch.abs(comp_src) <= tolerance
    zero_dst = torch.abs(comp_dst) <= tolerance
    rad = torch.where(
        zero_src & zero_dst, torch.tensor(0.),
        torch.where(
            zero_src, torch.angle(comp_dst),
            torch.where(
                zero_dst, -torch.angle(comp_src),
                torch.angle(comp_src) - torch.angle(comp_dst)
            )
        )
    )
    rad = torch.remainder(rad + offset, 2 * torch.pi) if wrap else rad + offset
    return torch.rad2deg(rad) if to_degree else rad

def inv_sigmoid(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))

class ReckUnitaryLayer(nn.Module):
    def __init__(self, dim=4, lp_db=0, lc_db=0, phis=torch.rand(6, dtype=torch.float64)*2*torch.pi, thetas=torch.rand(3, dtype=torch.float64)*2*torch.pi):
        super().__init__()
        self.dim = dim
        self.num = int(dim * (dim - 1) / 2)
        self.lp = torch.tensor(10 ** (lp_db / 10), dtype=torch.float64)
        self.lc = torch.tensor(10 ** (lc_db / 10), dtype=torch.float64)

        self.phis = nn.Parameter(phis.detach().clone())
        self.thetas = nn.Parameter(thetas.detach().clone())
        self.thetas_constant = torch.zeros(3, dtype=torch.float64, device=self.phis.device) # Buffer-like
        self.register_buffer("alphas", torch.zeros(self.dim, dtype=torch.float64))
        self.q_list, self.p_list = self.get_q_p()

    def get_q_p(self):
        q_list, p_list = [], []
        for p in range(1, self.dim):
            for q in range(p):
                q_list.append(q)
                p_list.append(p)
        return q_list, p_list
    
    def theta_concat(self):
        return torch.cat([self.thetas_constant.to(self.thetas.device), self.thetas])

    def reconstruct(self, phis, thetas, alphas):
        mat = torch.diag(torch.exp(-1j * alphas)).to(torch.complex128)
        for q, p, theta, phi in zip(self.q_list, self.p_list, thetas.flip(0), phis.flip(0)):
            mat = mat @ u2mzi(self.dim, q, p, theta, phi, lp=self.lp, lc=self.lc).conj().T
        return mat    

    def forward(self):
        return self.reconstruct(self.phis, self.theta_concat(), self.alphas)

class ThermalPhaseLayer(nn.Module):
    """
    Represents one column of Phase Shifters with physical constraints.
    Input dim is 3 (active heaters), Waveguide dim is 4.
    """
    def __init__(self, dim=3, bottom_heaters=True, r_diag = (0.99, 0.6, 0.1)):
        super().__init__()
        self.dim = dim
        self.bottom_heaters = bottom_heaters
        
        # Constrained Alpha Initialization (Crosstalk)
        r0 = inv_sigmoid(r_diag[0])
        r1 = inv_sigmoid(r_diag[1])
        r2 = inv_sigmoid(r_diag[2])
        self.raw_diag = nn.Parameter(torch.tensor([r0, r1, r2], dtype=torch.float64))
        self.raw_matrix = nn.Parameter(torch.full((3, 3), inv_sigmoid(0.4), dtype=torch.float64))
        
        # Physical parameters
        self.h_0 = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros((3,3), dtype=torch.float64))

    def get_constrained_alpha(self):
        sigmoid = torch.sigmoid
        a00 = 10.0 * sigmoid(self.raw_diag[0]) + 1e-6
        a11 = a00 * (0.5 + 0.5 * sigmoid(self.raw_diag[1]))
        a22 = a11 * (0.5 + 0.5 * sigmoid(self.raw_diag[2]))
        diag_values = torch.stack([a00, a11, a22])
        
        lower_bound = -4.0 
        upper_bound = a22 
        constrained_matrix = lower_bound + (upper_bound - lower_bound) * sigmoid(self.raw_matrix)
        
        eye = torch.eye(3, device=self.raw_diag.device, dtype=self.raw_diag.dtype)
        alpha = (1 - eye) * constrained_matrix + torch.diag_embed(diag_values)
        return alpha

    def forward(self, x):
        """
        x: Power tensor for this specific layer (Batch, 3)
        Returns: Diagonal Phase Matrix (Batch, 4, 4)
        """
        alpha = self.get_constrained_alpha()
        
        if self.bottom_heaters:
            alpha_val = alpha.to(x.device)
            beta_val = self.beta.to(x.device)
        else:
            alpha_val = alpha.to(x.device).flip([0,1])
            beta_val = self.beta.to(x.device).flip([0,1])
            
        h_0_val = self.h_0.to(x.device)
        
        h_list = h_0_val + (alpha_val @ x.T).T + (beta_val @ (x**2).T).T
        
        if self.bottom_heaters:
            h_diag = torch.cat([
                torch.ones((x.shape[0], 1), dtype=torch.complex128, device=x.device),
                torch.exp(1j * h_list),
            ], dim=1)
        else:
            h_diag = torch.cat([
                torch.exp(1j * h_list),
                torch.ones((x.shape[0], 1), dtype=torch.complex128, device=x.device)
            ], dim=1)
            
        return torch.diag_embed(h_diag)

class TriLayerStructure(nn.Module):
    def __init__(self, dim=4, lp_db=0, lc_db=0, r_diag=(0.99, 0.6, 0.1), phis_list =  [torch.rand(6, dtype=torch.float64) * 2 * torch.pi for _ in range(4)], thetas_list = [torch.rand(3, dtype=torch.float64)*2*torch.pi for _ in range(4)], init_fixed=False):
        super().__init__()
        self.dim = dim

        self.u1 = ReckUnitaryLayer(dim, lp_db, lc_db, phis_list[0], thetas_list[0])
        self.u2 = ReckUnitaryLayer(dim, lp_db, lc_db, phis_list[1], thetas_list[1])
        self.u3 = ReckUnitaryLayer(dim, lp_db, lc_db, phis_list[2], thetas_list[2])
        self.u4 = ReckUnitaryLayer(dim, lp_db, lc_db, phis_list[3], thetas_list[3]) # Final unitary
        

        self.p1 = ThermalPhaseLayer(dim=3, bottom_heaters=False, r_diag=r_diag)
        self.p2 = ThermalPhaseLayer(dim=3, bottom_heaters=False, r_diag=r_diag)
        self.p3 = ThermalPhaseLayer(dim=3, bottom_heaters=False, r_diag=r_diag)

        self.in_log_eff = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.out_log_eff = nn.Parameter(torch.zeros(dim, dtype=torch.float64))
        self.background = nn.Parameter(torch.full((dim,), 0.001, dtype=torch.float64))

        if init_fixed:
            self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            for u in [self.u1, self.u2, self.u3, self.u4]:
                u.phis.data.fill_(torch.pi / 4.0)
                u.thetas.data.fill_(0.0)

    def forward(self, x_full, indx):
        x1 = x_full[:, 0:3]
        x2 = x_full[:, 3:6]
        x3 = x_full[:, 6:9]

        U1 = self.u1()
        U2 = self.u2()
        U3 = self.u3()
        U4 = self.u4()

        Phi1 = self.p1(x1)
        Phi2 = self.p2(x2)
        Phi3 = self.p3(x3)

        in_gain = torch.diag(torch.exp(self.in_log_eff)).to(torch.complex128)
        out_gain = torch.diag(torch.exp(self.out_log_eff)).to(torch.complex128)
        
        chain = U4 @ Phi3 @ U3 @ Phi2 @ U2 @ Phi1 @ U1
        
        total_transfer = out_gain @ chain @ in_gain
        
        predict = torch.abs(total_transfer) ** 2
        predict = predict + nn.functional.softplus(self.background).unsqueeze(0).unsqueeze(-1)
        
        predict = torch.gather(predict, dim=2, index=indx)
        
        return predict.squeeze(2)


data = []
file_path = "full-structure-data"


layer_offsets = [5, 9, 14] 
heaters_per_layer = 3
num_input_channels = 4

for layer_idx, start_h in enumerate(layer_offsets):
    for ch in range(num_input_channels):
        for h_local in range(heaters_per_layer):
            
            file_h_idx = start_h + h_local 

            tensor_col_idx = (layer_idx * heaters_per_layer) + h_local
            
            fname = f"{file_path}/ch{ch + 1}_H{file_h_idx}.txt"
            
            try:
                with open(fname) as file:
                    for line in file:
                        try:
                            row = [float(num) for num in line.split()]
                            data.append([ch, tensor_col_idx] + row)
                        except:
                            pass
            except FileNotFoundError:
                print(f"Warning: File {fname} not found.")

df = pd.DataFrame(
    data, columns=['in_ch', 'heater_col_idx', 'current', 'ch1', 'ch2', 'ch3', 'ch4']
)

cols_to_normalize = ['ch1', 'ch2', 'ch3', 'ch4']
df[cols_to_normalize] = df[cols_to_normalize].div(df[cols_to_normalize].sum(axis=1), axis=0)


current_coefficient = 10**1
df["power"] = (current_coefficient * df["current"]) ** 2


total_heaters = 9
x_all = torch.zeros((len(df), total_heaters), dtype=torch.float64)

active_cols = torch.tensor(df["heater_col_idx"].values, dtype=torch.long)
powers = torch.tensor(df["power"].values, dtype=torch.float64)
x_all[torch.arange(len(df)), active_cols] = powers

target = torch.tensor(df[cols_to_normalize].values, dtype=torch.float64)

indx = torch.tensor(df["in_ch"].values, device=device)
indx = indx.unsqueeze(-1).unsqueeze(-1)
indx = indx.expand(-1, 4, -1) # Shape: (Batch, 4, 1)

print(f"Data Loaded. Samples: {len(df)}. Input Shape: {x_all.shape}")

model = TriLayerStructure().to(device)

n_starts = 1
best_loss = float('inf')
best_state = None

x_all = x_all.to(device)
target = target.to(device)
indx = indx.to(device)

run_log = []
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
log_path = f"run_log_{run_timestamp}.json"

for start in range(n_starts):
    print(f"\n=== Multistart {start + 1}/{n_starts} ===")

    if start == -1:
        model = TriLayerStructure().to(device)
    else:
        vals = torch.rand(3)
        vals, _ = vals.sort(descending=True)
        phis_list = [torch.tensor([0.08407550404831549, 4.609836125686209, 2.5394245981873294, 5.069458603371719, 5.776214109309022, 3.903089816463521]),
                     torch.tensor([0.02020515117377924, 2.505383942997596, 5.643790384657024, 5.538278061769468, 6.078201054812459, 3.5228895226752686]),
                     torch.tensor([2.9635496350322748, 3.484308226938264, 2.058556115736546, 5.094338217531042, 4.825690919817343, 4.001384148257002]),
                     torch.tensor([5.509775744418063, 5.173838453844745, 1.5460386689914973, 1.545159202206261, 5.916020133903857, 6.130278987861154])]

        thetas_list = [torch.tensor([1.2717759377740583, 5.934382161507198, 5.563051925158974]),
                       torch.tensor([1.2566983634218378, 4.781718545339196, 3.510183065184789]),
                       torch.tensor([6.231640852641818, 2.608245671636658, 6.146091752954586]),
                       torch.tensor([0.29691046649652486, 0.22239132233598058, 3.1022940920398434])]

        model = TriLayerStructure(r_diag=[0.6493263244628906, 0.6108959317207336, 0.2219613790512085], phis_list=phis_list, thetas_list=thetas_list).to(device)
        # with torch.no_grad():
        #     for u in [model.u1, model.u2, model.u3, model.u4]:
        #         u.phis.data += torch.randn_like(u.phis) * 0.3
        #         u.thetas.data += torch.randn_like(u.thetas) * 0.3
        #     for p in [model.p1, model.p2, model.p3]:
        #         p.h_0.data += torch.randn_like(p.h_0) * 0.1

    with torch.no_grad():
        init_alphas = [
            model.p1.get_constrained_alpha().cpu().numpy().diagonal(),
            model.p2.get_constrained_alpha().cpu().numpy().diagonal(),
            model.p3.get_constrained_alpha().cpu().numpy().diagonal(),
        ]

    optimizer_adam = optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=400, eta_min=0)

    print("Phase 1: Exploration (AdamW)...")
    pbar = tqdm(range(400))
    for step in pbar:
        optimizer_adam.zero_grad()
        pred = model(x_all, indx)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()
        scheduler.step()
    
        if step % 10 == 0:
            pbar.set_description(f"AdamW Loss: {loss.item():.6f}")

    optimizer_lbfgs = optim.LBFGS(
        model.parameters(), 
        lr=1,
        max_iter=50, 
        history_size=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-9
    )

    print("\nPhase 2: Fine-Tuning (LBFGS)...")
    def closure():
        optimizer_lbfgs.zero_grad()
        pred = model(x_all, indx)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        return loss

    pbar_lbfgs = tqdm(range(50))
    start_best = float('inf')

    for step in pbar_lbfgs:
        loss = optimizer_lbfgs.step(closure)
        pbar_lbfgs.set_description(f"LBFGS Loss: {loss.item():.8f}")
        
        if loss.item() < start_best:
            start_best = loss.item()

        if step > 20 and loss.item() > start_best * (1 - 1e-7):
            print("Converged.")
            break

    run_log.append({
        "start": start + 1,
        "alpha_initializing_values": vals.tolist(),
        "initial_phis_list, layer 1": phis_list[0].tolist(),
        "initial_phis_list, layer 2": phis_list[1].tolist(),
        "initial_phis_list, layer 3": phis_list[2].tolist(),
        "initial_phis_list, layer 4": phis_list[3].tolist(),
        "initial_thetas_list, layer 1": thetas_list[0].tolist(),
        "initial_thetas_list, layer 2": thetas_list[1].tolist(),
        "initial_thetas_list, layer 3": thetas_list[2].tolist(),
        "initial_thetas_list, layer 4": thetas_list[3].tolist(),
        "init_diag_p1": init_alphas[0].tolist(),
        "init_diag_p2": init_alphas[1].tolist(),
        "init_diag_p3": init_alphas[2].tolist(),
        "final_mse": start_best,
    })
    
    if start_best < best_loss:
        best_loss = start_best
        best_state = copy.deepcopy(model.state_dict())
    print(f"  Start {start + 1} best loss: {start_best:.8f}")

model.load_state_dict(best_state)

with open(log_path, "w") as f:
    json.dump(run_log, f, indent=2)
print(f"Log saved to {log_path}")

print(f"Final Best Loss: {best_loss}")
torch.save(model.state_dict(), "tri_layer_model.pth")


with torch.no_grad():
    print("\nAlpha Matrices per Layer:")
    print("Layer 1:\n", model.p1.get_constrained_alpha().cpu().numpy())
    print("Layer 2:\n", model.p2.get_constrained_alpha().cpu().numpy())
    print("Layer 3:\n", model.p3.get_constrained_alpha().cpu().numpy())

current_file = ipynbname.path() 
run_id = time.strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/{run_id}"
os.makedirs(log_dir)

shutil.copy(current_file, f"{log_dir}/script_snapshot.ipynb")

with open(f"{log_dir}/metrics.txt", "w") as f:
    f.write(f"min_loss: {best_loss}\n")
