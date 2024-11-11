import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import array
from qpsolvers import solve_qp
from numpy import linalg as LA
from timeit import default_timer as timer
import numpy as np
from numpy import linalg as LA

def point_projected(n: int, radius):
    """Generates the point to be projected
    Args:
        n (int): [The length of the point to be projected.]
    Returns:
        data (float): [The point to be projected, following the standard Normal distribution.]
    """    
    
    mu, sigma = radius/n, 1. 
    
    point_to_be_projected = np.abs(np.random.normal(mu,sigma,n))
    
    while LA.norm(point_to_be_projected, p) ** p <= radius:  
        mu *= 5
        point_to_be_projected = np.abs(np.random.normal(mu,sigma,n))

    return point_to_be_projected

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


def get_lp_ball_projection(starting_point, 
                    point_to_be_projected, 
                                        p, 
                                   radius, 
                                  epsilon,
                                      tol,
                                    model,
                                  tau=1.1,  
                                  MAX_ITER=100,**kwargs):

    # Step 1 and 2 in IRBP. 
    n = point_to_be_projected.shape[0]


    condition_right = 100    
    signum = np.sign(point_to_be_projected) 
    yAbs = signum * point_to_be_projected  # yAbs lies in the positive orthant of R^n

    lamb = 0.0
    residual_alpha0 = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
    residual_beta0 =  abs(LA.norm(starting_point, p) ** p - radius)

    cnt = 0  
    timeStart = timer()
    perf = []
    while True:

        cnt += 1
        
        alpha_res = (1. / n) * LA.norm((yAbs - starting_point) * starting_point - p * lamb * starting_point ** p, 1)
        beta_res = abs(LA.norm(starting_point, p) ** p - radius)

        if (alpha_res<tol and beta_res < tol) or cnt > MAX_ITER:
            timeEnd = timer()
            x_final = signum * starting_point # symmetric property of lp ball
            break

        # Step 3 in IRBP. Compute the weights
            
        if model == 1:
            weights = p * 1. / ((np.abs(starting_point) + epsilon) ** (1 - p) + 1e-12)
        else:
            u = epsilon ** (p-1);
            k = (p-1) * epsilon ** p
            ind = (starting_point >=  epsilon);
            weights = p * 1. / (np.abs(starting_point) ** (1 - p) + 1e-12)
            weights[~ind] = p * 1. / (epsilon ** (1 - p) + 1e-12)
        
        # Step 4 in IRBP. Solve the subproblem for x^{k+1}
        if model == 1:
            gamma_k = radius - LA.norm(abs(starting_point) + epsilon, p) ** p + np.inner(weights, abs(starting_point))
        #elif model == 2: 
        #    gamma_k = radius - LA.norm(abs(starting_point[ind]), p) ** p - (1 - p) * LA.norm(abs(starting_point[~ind]) + epsilon, p) ** p + np.inner(weights, starting_point)
        else:
            gamma_k = radius - LA.norm(abs(starting_point[ind]), p) ** p - np.sum(p * abs(starting_point[~ind]) * u - k) + np.inner(weights, starting_point)
    
        assert gamma_k > 0, "The current Gamma is non-positive"
        
        # Subproblem solver : The projection onto weighted l1-ball
        x_new, lamb = get_weighted_l1_ball_projection_sort(yAbs, weights, gamma_k)
        
        # Step 5 in IRBP. Set the new relaxation vector epsilon according to the proposed condition
        condition_left = LA.norm(x_new - starting_point, 2) * LA.norm(np.sign(x_new - starting_point) * weights, 2) ** tau

        if condition_left <= condition_right:
            if model == 1:
                theta = np.minimum(beta_res, 1. / np.sqrt(cnt))** (1. / p)
            else:
                theta = np.maximum(1e-6, np.minimum(beta_res, 1. / np.sqrt(cnt))** (1. / p))
            epsilon = theta * epsilon
    
        # Step 6 in IRBP. Set k <--- (k+1)
        starting_point = x_new.copy()
    return x_final, lamb, timeEnd-timeStart, cnt
    
# Initialization
pd.set_option('display.max_rows',None)
np.set_printoptions(threshold=np.inf)
#np.random.seed(10)
nn = np.array([1000])#([10,100,1000,10000,100000])
rr = np.array([1.,2.,4.,8.,16.,32.,64.,128.])
rr = np.array([8.])

tol = 1e-8
p = 0.6

result = pd.DataFrame(columns=['n','radius', 'time1','time2','iteration1','iteration2'])
#result = pd.DataFrame(columns=['n','radius', 'time1','time2'])

for data_dim in nn:
    for radius in rr:
        time1 = 0.
        time2 = 0.
        iter1 = 0
        iter2 = 0
        point_to_be_projected = point_projected(data_dim, radius) - 3.
        data_record = [point_to_be_projected]
        tot = 1
        for k in range(tot):
            x_ini = np.zeros(data_dim, dtype=np.float64)
            #rand_num = np.random.uniform(0., 1., data_dim)
            #rand_num_norm = LA.norm(np.ones(data_dim), ord=1)
            epsilon_ini = 0.4 * (1 * radius / data_dim) ** (1. / p) # ensure that the point is feasible.
            
            x_irbp, dual, runningTime1, iteration1 = get_lp_ball_projection(x_ini, 
                                                         point_to_be_projected, 
                                                                             p, 
                                                                        radius, 
                                                                   epsilon_ini,
                                                                           tol,
                                                                             1,
                                                                       tau=1.1,  
                                                                 MAX_ITER=1000)

            x_erbp, dual, runningTime2, iteration2 = get_lp_ball_projection(x_ini, 
                                                         point_to_be_projected, 
                                                                             p, 
                                                                        radius, 
                                                                   epsilon_ini,
                                                                           tol,
                                                                             2,
                                                                       tau=1.1,  
                                                                 MAX_ITER=1000)
            
            time1 += runningTime1
            time2 += runningTime2
            iter1 += iteration1
            iter2 += iteration2
            point_to_be_projected = point_projected(data_dim, radius)
            data_record = np.append(data_record, [point_to_be_projected], axis=0)

        result = result.append({'n':data_dim, 'radius':radius, 'time1':time1/tot, 'time2':time2/tot, 'iteration1':iter1/tot, 'iteration2':iter2/tot}, ignore_index = True)
        pd.DataFrame(data_record[:-1]).to_csv('lp_projected_data_lu_{}_{}_{}.csv'.format(data_dim, radius, p))