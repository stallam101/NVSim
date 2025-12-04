import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import lognorm, pareto, genpareto, weibull_min, fisk, gamma

print("🔍 Calculating Gamma Distribution Parameters")
print("=" * 50)

target_mean_s = ((0.13)*10**6)/(200)
k_s = 0.13
scale_for_mean_s = target_mean_s / (k_s*8)
target_mean_r = ((0.11)*10**6)/200
k_r = 0.13
scale_for_mean_r = target_mean_r / (k_r*10)

print(f"SET calculations:")
print(f"  target_mean_s = {target_mean_s}")
print(f"  k_s = {k_s}")
print(f"  scale_for_mean_s = {scale_for_mean_s}")
print()
print(f"RESET calculations:")
print(f"  target_mean_r = {target_mean_r}")
print(f"  k_r = {k_r}")
print(f"  scale_for_mean_r = {scale_for_mean_r}")
print()

# Calculate loc parameters
loc_013_13 = target_mean_r - k_s*scale_for_mean_r
loc_013_11 = target_mean_s - k_r*scale_for_mean_s

print(f"Distribution Parameters:")
print(f"gamma_k=0.013_mu=0.13: a={k_r}, scale={scale_for_mean_r}, loc={loc_013_13}")
print(f"gamma_k=0.013_mu=0.11: a={k_s}, scale={scale_for_mean_s}, loc={loc_013_11}")
print()

HEAVY_DISTS = {
    "gamma_k=0.013_mu=0.13":  gamma(a=k_r, scale=scale_for_mean_r, loc=target_mean_r - k_s*scale_for_mean_r),
    "gamma_k=0.013_mu=0.11":  gamma(a=k_s, scale=scale_for_mean_s, loc=target_mean_s - k_r*scale_for_mean_s)
}

print("📊 Final Distribution Objects:")
for name, dist in HEAVY_DISTS.items():
    print(f"{name}:")
    print(f"  a (shape) = {dist.args[0] if dist.args else 'N/A'}")
    print(f"  scale = {dist.kwds.get('scale', 1.0)}")
    print(f"  loc = {dist.kwds.get('loc', 0.0)}")
    print()

print("🎯 FOR NVSIM CELL FILE UPDATE:")
print("SET Distribution (gamma_k=0.013_mu=0.13):")
print(f"  -SetGammaK: {k_r}")
print(f"  -SetGammaTheta: {scale_for_mean_r}")  
print(f"  -SetGammaLoc: {loc_013_13}")
print()
print("RESET Distribution (gamma_k=0.013_mu=0.11):")
print(f"  -ResetGammaK: {k_s}")
print(f"  -ResetGammaTheta: {scale_for_mean_s}")
print(f"  -ResetGammaLoc: {loc_013_11}")
print()
def generate_samples(base_dist, n_iid=100, n_reps=20, n_outputs=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    results = np.zeros(n_outputs)
    for i in range(n_outputs):
        reps = []
        for _ in range(n_reps):
            samples = base_dist.rvs(size=n_iid, random_state=rng)
            reps.append(np.max(samples))
        results[i] = np.min(reps)
    return results
fig, ax = plt.subplots(1, 1, figsize=(7, 18))
for idx, (name, dist) in enumerate(HEAVY_DISTS.items()):
    data = (generate_samples(dist, n_iid=21, n_reps=40, n_outputs=100, random_state=idx))*200/10**6 + 0.55
    sns.histplot(
        data,
        stat="probability",
        alpha=0.75,
        ax=ax,
        binwidth=2*8000/(5*(10**6))
    )
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_xlim([0.54, 0.90])
    #ax.set_xlim([0.65, 0.67])
    ax.set_ylabel("density")
    ax.set_yscale('log')
    ax.set_ylim([10**-4, 10**0])
data = generate_samples(
    HEAVY_DISTS["gamma_k=0.013_mu=0.13"], n_iid=3, n_reps=40, n_outputs=10000, random_state=None
) +  generate_samples(
    HEAVY_DISTS["gamma_k=0.013_mu=0.11"], n_iid=3, n_reps=40, n_outputs=10000, random_state=None
) + 0.54
sns.histplot(
    data,
    bins=40,
    stat="probability",
    alpha=0.75,
    ax=ax,
    binwidth=1*8000/(5*(10**6))
)
ax.set_title(name)
ax.set_xlabel("value")
ax.set_xlim([0.54, 0.90])
ax.set_ylabel("density")
ax.set_yscale('log')
ax.set_ylim([10**-4, 10**0])
plt.tight_layout()
plt.show()