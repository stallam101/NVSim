import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Gamma distribution parameters for NVSim simulation
target_mean_s = ((0.13)*10**6)/(200)
target_mean_r = ((0.11)*10**6)/200
k_s = k_r = 0.13
scale_for_mean_s = target_mean_s / (k_s*8)
scale_for_mean_r = target_mean_r / (k_r*10)

# Create distributions matching NVSim cell file parameters
HEAVY_DISTS = {
    "gamma_k=0.013_mu=0.13":  gamma(a=k_r, scale=scale_for_mean_r, loc=target_mean_r - k_s*scale_for_mean_r),
    "gamma_k=0.013_mu=0.11":  gamma(a=k_s, scale=scale_for_mean_s, loc=target_mean_s - k_r*scale_for_mean_s)
}
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