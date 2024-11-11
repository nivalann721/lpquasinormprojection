import numpy as np
from numpy import linalg as LA
from timeit import default_timer as timer


def get_weighted_l1_ball_projection_sort(y, w, a):
    
    length = len(y)
    z = np.zeros(length, dtype=np.float64)
    x_opt = np.zeros(length, dtype=np.float64)
    z_perm =  np.zeros(length, dtype=np.int32)

    z = y / w
    for i in range(length):
        z_perm[i] = i
    z_perm = np.argsort(-z)

    i = 0
    sumWY = w[z_perm[i]] * y[z_perm[i]]
    Ws = w[z_perm[i]] * w[z_perm[i]]
    tau = (sumWY - a) / Ws
    for i in range(1, length):
        if z[z_perm[i]] < tau: break
        sumWY += w[z_perm[i]] * y[z_perm[i]]
        Ws += w[z_perm[i]] * w[z_perm[i]]
        tau = (sumWY - a) / Ws

    x_opt = y - w * tau
    return np.maximum(x_opt, 0.0), tau

def get_lp_ball_projection_irbp1(
    starting_point,
    point_to_be_projected,
    p,
    radius,
    epsilon,
    Tau=1.1,
    condition_right=100,
    tol=1e-8,
    MAX_ITER=1000,
    MAX_TIME=200,
    **kwargs
):
    if LA.norm(point_to_be_projected, p) ** p <= radius:
        return point_to_be_projected, 0.0, 0

    # Step 1 and 2 in IRBP.
    n = point_to_be_projected.shape[0]

    signum = np.sign(point_to_be_projected)
    yAbs = signum * point_to_be_projected  # yAbs lives in the positive orthant of R^n

    lamb = 0.0
    residual_alpha0 = (1.0 / n) * LA.norm(
        (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
    )
    residual_beta0 = abs(LA.norm(starting_point, p) ** p - radius)

    cnt = 0

    # The loop of IRBP
    timeStart = timer()
    while True:

        cnt += 1
        alpha_res = (1.0 / n) * LA.norm(
            (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
        )
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)

        if (
            max(alpha_res, beta_res)
            < tol
            or cnt > MAX_ITER
            or timer() - timeStart > MAX_TIME
        ):
            timeEnd = timer()
            x_final = signum * starting_point  # symmetric property of lp ball
            break

        # Step 3 in IRBP. Compute the weights
        weights = p * 1.0 / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)

        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = (
            radius
            - LA.norm(abs(starting_point) + epsilon, p) ** p
            + np.inner(weights, np.abs(starting_point))
        )

        assert gamma_k > 0, "The current Gamma is non-positive"

        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weighted_l1_ball_projection_sort(yAbs, weights, gamma_k)
        x_new[np.isnan(x_new)] = np.zeros_like(x_new[np.isnan(x_new)])

        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = (
            LA.norm(x_new - starting_point, 2)
            * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** Tau
        )

        if condition_left <= condition_right:
            theta = np.minimum(beta_res, 1.0 / np.sqrt(cnt)) ** (1.0 / p)
            epsilon = theta * epsilon

        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()
    return x_final, timeEnd - timeStart, cnt

def get_lp_ball_projection_irbp2(
    starting_point,
    point_to_be_projected,
    p,
    radius,
    epsilon,
    Tau=1.1,
    condition_right=100,
    tol=1e-8,
    MAX_ITER=1000,
    MAX_TIME=200,
    **kwargs
):

    if LA.norm(point_to_be_projected, p) ** p <= radius:
        return point_to_be_projected, 0.0, 0

    # Step 1 and 2 in IRBP.
    n = point_to_be_projected.shape[0]

    signum = np.sign(point_to_be_projected)
    yAbs = signum * point_to_be_projected 

    lamb = 0.0
    residual_alpha0 = (1.0 / n) * LA.norm(
        (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
    )
    residual_beta0 = abs(LA.norm(starting_point, p) ** p - radius)

    cnt = 0

    # The loop of IRBP
    timeStart = timer()
    while True:

        cnt += 1
        u = epsilon ** (p-1);
        k = (p-1) * epsilon ** p
        ind = (np.abs(starting_point) >=  epsilon);
        alpha_res = (1.0 / n) * LA.norm(
            (yAbs - starting_point) * starting_point - p * lamb * starting_point**p, 1
        )
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)

        if (
            max(alpha_res, beta_res)
            < tol
            or cnt > MAX_ITER
            or timer() - timeStart > MAX_TIME
        ):
            timeEnd = timer()
            x_final = signum * starting_point  # symmetric property of lp ball
            break

        # Step 3 in IRBP. Compute the weights
        weights = p * 1. / (np.abs(starting_point) ** (1 - p) + 1e-12)
        weights[~ind] = p * 1. / (epsilon ** (1 - p) + 1e-12)

        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        gamma_k = (
            radius - LA.norm(abs(starting_point[ind]), p) ** p 
            - np.sum(p * abs(starting_point[~ind]) * u - k)
            + np.inner(weights, starting_point)
        )

        assert gamma_k > 0, "The current Gamma is non-positive"

        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weighted_l1_ball_projection_sort(yAbs, weights, gamma_k)
        x_new[np.isnan(x_new)] = np.zeros_like(x_new[np.isnan(x_new)])

        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = (
            LA.norm(x_new - starting_point, 2)
            * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** Tau
        )

        if condition_left <= condition_right:
            theta = np.maximum(1e-6, np.minimum(beta_res, 1.0 / np.sqrt(cnt))) ** (1.0 / p)
            epsilon = theta * epsilon

        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()
    return x_final, timeEnd - timeStart, cnt


def PGD_Lp(signal, i, m, p, algorithm, maxtime=100):
    """ """
    Iter_max = int(1e3)
    x = signal[:, i]
    A = np.random.normal(0.0, 1.0, [m, x.shape[0]])
    y = A @ x
    gradf = lambda x: A.T @ (A @ x - y)
    radius = LA.norm(x, p) ** p
    if radius < 1e-4:  # If all 0, just return
        return np.zeros_like(x)
    Lf = np.max(LA.eigvals(A.T @ A).real)
    mu = 1 / Lf
    
    n = len(x)
    t0 = timer()
    x = np.zeros_like(x)
    for i in range(Iter_max):
        grad = gradf(x)
        z = x - mu * grad
        x_pre = x

        rand_num = np.random.uniform(0, 1)
        abs_norm = n * rand_num
        epsilon = 0.9 * (rand_num * radius / abs_norm) ** (1/p)  # ensure that the point is feasible.
        #x, lamb = WeightLpBallProjection(n, np.zeros(n), z, p, radius, epsilon)
        if algorithm == 'irbp_1':
            x, _, _ = get_lp_ball_projection_irbp1(np.zeros(n), z, p, radius, epsilon)
        else:
            x, _, _ = get_lp_ball_projection_irbp2(np.zeros(n), z, p, radius, epsilon)
        if LA.norm(x - x_pre) < 1e-4 or timer() - t0 > maxtime:
            break
    return x

import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import cv2
import pywt
import os
import warnings
from skimage.metrics import peak_signal_noise_ratio

# Import the baseline file
sys.path.append("..")
warnings.filterwarnings("ignore")


def psnr(gt, pred):
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max() - gt.min())


image_path = f"./image/04.png"
output_image_repo = f"./result"
wl = "haar"  # wavelet transform type
level = 4  # wavelet transform level
eps = 0.04  # wavelet transform is not completely sparse, ignore value less than eps


if not os.path.isdir(output_image_repo):
    os.makedirs(output_image_repo)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) * 1.0 / 255.0
coeffs = pywt.wavedec2(image, wavelet=wl, level=level)
signal_raw, coeff_slices = pywt.coeffs_to_array(coeffs)
signal = np.zeros_like(signal_raw)
signal[np.abs(signal_raw) > eps] = signal_raw[np.abs(signal_raw) > eps]

p = 0.6  # lp ###
m = 200
t0 = timer()
signal_recovered = np.zeros_like(signal)
algorithm = 'irbp_1'
for i in range(256):
    signal_recovered[:, i] = PGD_Lp(signal, i, m, p, algorithm) ##

coeffs_from_arr = pywt.array_to_coeffs(
    signal_recovered, coeff_slices, output_format="wavedec2"
)
image_recovery = pywt.waverec2(coeffs_from_arr, wavelet=wl)
cv2.imwrite(output_image_repo + f"/image1.jpg", image_recovery * 255.0)
print(algorithm, timer() - t0, psnr(image, image_recovery))

t0 = timer()
signal_recovered = np.zeros_like(signal)
algorithm = 'irbp_2'
for i in range(256):
    signal_recovered[:, i] = PGD_Lp(signal, i, m, p, algorithm) ##

coeffs_from_arr = pywt.array_to_coeffs(
    signal_recovered, coeff_slices, output_format="wavedec2"
)
image_recovery = pywt.waverec2(coeffs_from_arr, wavelet=wl)
cv2.imwrite(output_image_repo + f"/image2.jpg", image_recovery * 255.0)
print(algorithm, timer() - t0, psnr(image, image_recovery))

