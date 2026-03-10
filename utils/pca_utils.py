import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import einops
import torch

def get_pca_components(psfs_all, pca_n=100):
    print("PCA: Starting", psfs_all.shape)
    # print(f"NaN values {np.isnan(psfs_all).sum()}")
    inc, outc, h, w, hp, wp = psfs_all.shape

    basis_psfs = np.zeros((inc, outc, pca_n + 1, hp, wp), dtype='float32')
    basis_coef = np.zeros((inc, outc, pca_n + 1, h, w), dtype='float32')
    pca_var = np.zeros((inc, outc, pca_n), dtype='float32')

    # Note: last basis function stores the mean. so PCA vector is always [..., 1] because we need to add mean
    for i in range(inc):
        for j in range(outc):
            pca = PCA(n_components=pca_n)
            psfs_c = np.copy(psfs_all[i, j]).reshape((h * w, -1))

            psfs_c_mean = psfs_c.mean(0, keepdims=True)
            # psfs_c_var = ((psfs_c - psfs_c_mean) ** 2).sum(0) / (h * w)
            # psfs_c_var[psfs_c_var == 0] = 1

            # plt.plot(psfs_c_mean[0])
            # plt.show()
            #
            # plt.semilogy(psfs_c_var)
            # plt.show()

            psfs_c -= psfs_c_mean
            #             psfs_c /= psfs_c_var**0.5

            # plt.plot((psfs_c ** 2).sum(0) / (h * w))
            # plt.show()

            with threadpool_limits(limits=1):
                pca.fit(psfs_c)
            psf_c_transformed = pca.transform(psfs_c)

            basis_coef[i, j, :pca_n] = psf_c_transformed.reshape((h, w, pca_n)).transpose([2, 0, 1])
            basis_coef[i, j, pca_n] = 1.0  # always add mean

            basis_psfs[i, j, pca_n] = psfs_c_mean.reshape((hp, wp))
            # basis_psfs[i, j, pca_n + 1] = psfs_c_var.reshape((hp, wp))
            basis_psfs[i, j, :pca_n] = pca.components_.reshape((-1, hp, wp))

            # plt.plot(pca.mean_)
            # plt.show()

            pca_var[i, j] = pca.explained_variance_

    pca_components = np.copy(basis_psfs[:, :, :pca_n])
    pca_mean = np.copy(basis_psfs[:, :, pca_n])

    if pca_n == 1:  # all information in the mean. drop the first PCA component
        basis_psfs = basis_psfs[:, :, 1:]
        basis_coef = basis_coef[:, :, 1:]

    print("PCA done")

    return basis_psfs, basis_coef, pca_components, pca_mean, pca_var

def get_pca_components_torch(psfs_all, pca_n=100):
    inc, outc, h, w, H, W = psfs_all.shape
    A = einops.rearrange(psfs_all, 'a b h w H W -> (a b) (h w) (H W)')
    center = A.mean((0, 1), keepdim=True) # 1 1 n
    A -= center
    basis_coef = torch.ones((A.shape[0], A.shape[1], pca_n+1), dtype=torch.float32, device=A.device)
    basis_psfs = torch.zeros((A.shape[0], pca_n+1, A.shape[2]), dtype=torch.float32, device=A.device)
    for i in range(len(A)):
        U,S,V = torch.pca_lowrank(A[i], q=pca_n, niter=2, center=False)
        #       A: mxn U: mxq S: q V: nxq
        # Note: A    = U      S    (V^H) 
        #       AV = US(V^H)V = US
        basis_coef[i, :, :-1] = U
        basis_psfs[i, :-1, :] = torch.matmul(S, V.H)
    basis_psfs[:, -1, :] = center[0,0]
    basis_coef = einops.rearrange(basis_coef, '(a b) (h w) q -> a b q h w', a=inc, b=outc, h=h, w=w)
    basis_psfs = einops.rearrange(basis_psfs, '(a b) q (H W) -> a b q H W', a=inc, b=outc, H=H, W=W)
    return basis_psfs, basis_coef



