import torch


def mmd(X1, X2, bandwidth='median'):
    """Compute Maximum Mean Discrepancy"""
    if len(X1.shape) == 4:
        X1 = X1.view(len(X1), -1)
    if len(X2.shape) == 4:
        X2 = X2.view(len(X2), -1)

    N1 = len(X1)
    X1_sq = X1.pow(2).sum(1).unsqueeze(0)
    X1_cr = torch.mm(X1, X1.t())
    X1_dist = X1_sq + X1_sq.t() - 2 * X1_cr

    N2 = len(X2)
    X2_sq = X2.pow(2).sum(1).unsqueeze(0)
    X2_cr = torch.mm(X2, X2.t())
    X2_dist = X2_sq + X2_sq.t() - 2 * X2_cr

    X12 = torch.mm(X1, X2.t())
    X12_dist = X1_sq.t() + X2_sq - 2 * X12

    # median heuristic to select bandwidth
    if bandwidth == 'median':
        X1_triu = X1_dist[torch.triu(torch.ones_like(X1_dist), diagonal=1) == 1]
        bandwidth1 = torch.median(X1_triu)
        X2_triu = X2_dist[torch.triu(torch.ones_like(X2_dist), diagonal=1) == 1]
        bandwidth2 = torch.median(X2_triu)
        bandwidth_sq = ((bandwidth1 + bandwidth2) / 2).detach()
    else:
        bandwidth_sq = (bandwidth ** 2)

    C = - 0.5 / bandwidth_sq
    K11 = torch.exp(C * X1_dist)
    K22 = torch.exp(C * X2_dist)
    K12 = torch.exp(C * X12_dist)
    K11 = (1 - torch.eye(N1).to(X1.device)) * K11
    K22 = (1 - torch.eye(N2).to(X1.device)) * K22
    mmd = K11.sum() / N1 / (N1 - 1) + K22.sum() / N2 / (N2 - 1) - 2 * K12.mean()
    return mmd


