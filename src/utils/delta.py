#CODE COPIED AND ADJUSTED FROM https://github.com/htdt/hyp_metric/blob/master/delta.py

import torch

def delta_hyp(dismat): # Takes in a distance matrix
    p = 0
    row = dismat[p, :][None, :]
    col = dismat[:, p][:, None]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = torch.minimum(XY_p[:, :, None], XY_p[None, :, :]).max(1).values
    return (maxmin - XY_p).max()


def calc_hyperbolicity(emb):
    #if torch.all(emb == 0):
    #    print("All embedding vectors are zeros.")
    #elif torch.any(emb == 0):
    #    print("Some embedding vectors contain zeros.")
    #else:
    #    print("No embedding vectors are entirely zeros.")
    result = []
    for _ in range(100):
        if len(emb) >= 2000:
            idx = torch.randperm(len(emb))[:2000]
        else:
            idx = torch.randperm(len(emb))[:len(emb)]
        emb_cur = emb[idx]
        dists = torch.cdist(emb_cur, emb_cur)
        # Assuming 'dists' is your distance matrix
        #if torch.isinf(dists).any():
        #    print("The distance matrix contains infinite values.")

        #if torch.isnan(dists).any():
        #    print("The distance matrix contains NaN values.")
        delta = delta_hyp(dists)
        diam = dists.max()
        rel_delta = (2 * delta) / diam
        result.append(rel_delta)
    rel_delta_mean = torch.tensor(result).mean().item()
    c = (0.144 / rel_delta_mean) ** 2
    #print(f"δ = {rel_delta_mean:.3f}, c = {c:.3f}")
    return rel_delta_mean, c
    
