"""
schisto_ca_model.py
--------------------
Cellular-automaton (CA) implementation of a spatial compartmental model
for schistosomiasis transmission.

Formal definition (following the CA formalism A = <G, Z, N, f>):
    G : 2D square lattice of size L x L. Each cell g in G represents a
        human/snail micro-habitat (a water-contact site + surrounding
        household cluster).
    Z : cell state = (Sh, Ih, Rh, Ss, Is) -- fractions (in [0,1], summing
        to 1 within each species) of Susceptible / Infected / Recovered
        humans and Susceptible / Infected snails at that cell.
    N : Moore neighbourhood (8 neighbours) -- represents shared water
        bodies / hydrological connectivity between adjacent sites.
    f : transition function combining a local ODE (compartmental) step
        with a neighbourhood-diffusion step for the free-living larval
        stages (miracidia / cercariae), giving the automaton memory
        (Z^{|N|+1} -> Z form, since neighbours' PAST infected fractions
        feed forward).

Author: generated with Claude (Anthropic) assistance -- see prompt log
        linked in the accompanying report.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Parameters (illustrative, order-of-magnitude consistent with
#    published Schistosoma mansoni transmission models)
# ----------------------------------------------------------------------
L = 40                 # lattice side length -> G has L*L cells
STEPS = 180             # number of CA iterations (~ days)

beta_h = 0.18            # human infection rate per unit cercarial exposure
beta_s = 0.12            # snail infection rate per unit miracidial exposure
gamma_h = 1 / 90         # human "recovery" (natural + treatment clearance) rate
omega_h = 1 / 240        # human loss-of-immunity rate (Rh -> Sh)
mu_s = 1 / 60            # snail turnover rate (Is -> Ss, population renewal)
diff = 0.25              # larval diffusion coefficient across Moore neighbourhood
mda_step = 90            # simulated Mass Drug Administration (MDA) event (day)
mda_coverage = 0.6       # fraction of infected humans cured at MDA

# ----------------------------------------------------------------------
# 2. State arrays  Z: (Sh, Ih, Rh) for humans and (Ss, Is) for snails
# ----------------------------------------------------------------------
Sh = np.full((L, L), 0.97)
Ih = np.full((L, L), 0.03)
Rh = np.zeros((L, L))

Ss = np.full((L, L), 0.90)
Is = np.full((L, L), 0.10)

# seed a focal transmission hotspot near a water body (e.g. irrigation canal)
Ih[15:20, 15:25] = 0.35
Sh[15:20, 15:25] = 0.65
Is[15:20, 15:25] = 0.45
Ss[15:20, 15:25] = 0.55

kernel = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=float)
kernel /= kernel.sum()


def moore_average(field):
    """Diffuse a field over the Moore neighbourhood N (periodic boundary)."""
    padded = np.pad(field, 1, mode="wrap")
    out = np.zeros_like(field)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            out += padded[1 + di:1 + di + field.shape[0],
                          1 + dj:1 + dj + field.shape[1]]
    return out / 8.0


history = {"prev_h": [], "prev_s": []}
snapshots = {}

for t in range(STEPS):
    # local force of infection, mixed with neighbours' infected fraction
    # (this is the f transition function: local ODE step + N-coupling)
    lambda_h = beta_h * (0.7 * Is + 0.3 * moore_average(Is))   # human FOI from cercariae
    lambda_s = beta_s * (0.7 * Ih + 0.3 * moore_average(Ih))   # snail FOI from miracidia

    new_inf_h = lambda_h * Sh
    new_inf_s = lambda_s * Ss

    Sh_next = Sh - new_inf_h + omega_h * Rh
    Ih_next = Ih + new_inf_h - gamma_h * Ih
    Rh_next = Rh + gamma_h * Ih - omega_h * Rh

    Ss_next = Ss - new_inf_s + mu_s * Is
    Is_next = Is + new_inf_s - mu_s * Is

    # simulated MDA campaign: treat a fixed coverage of infected humans
    if t == mda_step:
        cured = mda_coverage * Ih_next
        Ih_next -= cured
        Rh_next += cured

    Sh, Ih, Rh = np.clip(Sh_next, 0, 1), np.clip(Ih_next, 0, 1), np.clip(Rh_next, 0, 1)
    Ss, Is = np.clip(Ss_next, 0, 1), np.clip(Is_next, 0, 1)

    # renormalise each species to sum to 1 per cell (closed local population)
    tot_h = Sh + Ih + Rh
    Sh, Ih, Rh = Sh / tot_h, Ih / tot_h, Rh / tot_h
    tot_s = Ss + Is
    Ss, Is = Ss / tot_s, Is / tot_s

    history["prev_h"].append(Ih.mean())
    history["prev_s"].append(Is.mean())

    if t in (0, mda_step - 1, STEPS - 1):
        snapshots[t] = Ih.copy()

# ----------------------------------------------------------------------
# 3. Output figure: spatial snapshots + prevalence time series
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, t in zip(axes[:3], sorted(snapshots.keys())):
    im = ax.imshow(snapshots[t], vmin=0, vmax=0.5, cmap="inferno")
    ax.set_title(f"Human infection prevalence, t = {t} d")
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes[2], fraction=0.046)

axes[3].plot(history["prev_h"], label="Human prevalence (Ih)", color="crimson")
axes[3].plot(history["prev_s"], label="Snail prevalence (Is)", color="steelblue")
axes[3].axvline(mda_step, color="grey", linestyle="--", label="MDA campaign")
axes[3].set_xlabel("time (days)")
axes[3].set_ylabel("mean prevalence")
axes[3].legend(fontsize=8)
axes[3].set_title("Mean-field prevalence")

plt.tight_layout()
plt.savefig("schisto_ca_results.png", dpi=180)
print("Saved figure to schisto_ca_results.png")
print(f"Final human prevalence: {history['prev_h'][-1]:.3f}")
print(f"Final snail prevalence: {history['prev_s'][-1]:.3f}")
