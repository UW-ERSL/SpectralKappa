import numpy as np


def build_1d_poisson(m: int, function_type="uniform"):
    """Build the 1D Poisson system for a given m (N = 2^m grid points)."""
    N = 2**m
    h = 1.0 / (N + 1)
    A = (np.diag(np.full(N, 2.0))
         + np.diag(np.full(N - 1, -1.0), k=1)
         + np.diag(np.full(N - 1, -1.0), k=-1))
    A /= h**2  # Scale by grid spacing squared
    if function_type == "uniform":
        b = np.ones(N) / np.sqrt(N)     # uniform load, unit norm
    elif function_type == "delta":
        b = np.zeros(N)
        b[N // 2] = 1.0
    elif function_type == "random":
        b = np.random.rand(N)
        b /= np.linalg.norm(b)          # random load, unit norm
    elif function_type == "sine":
        K = 1
        x = np.linspace(0, 1, N)
        b = np.sin(K*np.pi * x) 
        b /= np.linalg.norm(b)          # sine load, unit norm
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    return A, b

def build_2d_poisson(m, function_type="uniform"):
    N1 = 2**m
    h  = 1.0 / (N1 + 1)
    # 1D tridiagonal
    T  = (2.0 * np.eye(N1) - np.diag(np.ones(N1-1), 1)
                            - np.diag(np.ones(N1-1), -1)) / h**2
    I  = np.eye(N1)
    A  = np.kron(T, I) + np.kron(I, T)   # N^2 x N^2

    N = A.shape[0]
    if function_type == "uniform":
        b = np.ones(N) / np.sqrt(N)     # uniform load, unit norm
    elif function_type == "delta":
        b = np.zeros(N)
        mid = N1 // 2 * N1 + N1 // 2   # centre of N1 x N1 grid
        b[mid] = 1.0
    elif function_type == "random":
        b = np.random.rand(N)
        b /= np.linalg.norm(b)          # random load, unit norm
    elif function_type == "sine":
        K = 1
        x = np.linspace(0, 1, N)
        b = np.sin(K*np.pi * x) 
        b /= np.linalg.norm(b)          # sine load, unit norm
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    return A, b

def eigs_1d_poisson(m):
    N1  = 2**m
    h   = 1.0 / (N1 + 1)
    k   = np.arange(1, N1 + 1)
    lam = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    lam = np.sort(lam)
    return lam

def eigs_2d_poisson(m):
    N1   = 2**m
    h    = 1.0 / (N1 + 1)
    k    = np.arange(1, N1 + 1)
    lam1 = 4.0 / h**2 * np.sin(k * np.pi / (2*(N1+1)))**2
    # tensor product: all pairwise sums
    lam2d = (lam1[:, None] + lam1[None, :]).ravel()
    return lam2d

if __name__ == "__main__":
    m = 4
    A, b = build_1d_poisson(m, function_type="uniform")
    print("1D Poisson eigenvalues:", eigs_1d_poisson(m))

 