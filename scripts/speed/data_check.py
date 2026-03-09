#!/usr/bin/env python3

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from shared_code.fun_dfcspeed import dfc_speed_split, dfc_stream2fcd, ts2dfc_stream

# %%
# Load data HERE CHANGE THE PATH TO YOUR DATA
data = loadmat("/home/samy/Bureau/vscode/net_fluidity/scripts/dfc/test.mat")["TS"]

# sample frequencies in Hz
fr = 1 / 0.72 # HERE CHANGE THE SAMPLING FREQUENCY
sf = (fr * np.arange(55, 65)).astype(int)

# %% Compute dFC
speed_pool = []
for s in sf:
    # lag in number of frames
    lag = max(1, int(round(s * 0.005)))
    dfc_stream = ts2dfc_stream(data, s, lag, format_data="2D")
    fcd = dfc_stream2fcd(dfc_stream)
    speed = dfc_speed_split(
        dfc_stream,
        vstep=int(lag),
        tau_range=np.arange(2),
        method="pearson",
        return_fc2=False,
        time_offset=s,
    )

    print(
        s,
        lag,
        dfc_stream.min(),
        dfc_stream.max(),
        speed.min(),
        speed.max(),
        fcd.min(),
        fcd.max(),
    )
    speed_pool.append(speed)

# %%
tril_idx = np.tril_indices(data.shape[1], k=-1)

from shared_code.fun_optimization import fast_corrcoef

for i, k in enumerate(range(200)):
    wstart = k * lag
    wstop = wstart + s
    window = data[wstart:wstop, :]
    fc = fast_corrcoef(window)
    print(i, wstart, wstop, np.shape(window), np.shape(fc))

    dfc_stream[:, k] = fc[tril_idx]

# %%
# %%
# imshow of results
plt.imshow(dfc_stream, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

# %%
speed_flat = np.array([x.flatten() for x in speed_pool], dtype=object)
speed_flat = np.concatenate(speed_flat)


# plot hist of speed
plt.hist(speed_flat.ravel(), bins=100, histtype="step", density=True)
# %%
plt.imshow(fcd, aspect="auto", cmap="jet", vmin=0, vmax=1)
plt.colorbar()

# %%
