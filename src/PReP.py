import numpy as np
import scipy.sparse as sp
import datetime
from scipy.special import gamma
from joblib import Parallel, delayed
import multiprocessing
from scipy.optimize import minimize, minimize_scalar
from tools import deprecated
from random import shuffle
from src.GradDescent import gradient_descent
import sys


class PReP:

    delta_D = 1.e-50  # for values neighboring 0 due to Dirichlet distribution; lower bound for valid log, div
    delta_E = 0.0001  # for values neighboring 0 due to exponential distribution; lower bound for valid log, div
    sigma_lower = 0.01
    eta = 1.e-6  # converge condition
    nll_inc_tol = 0.01  # tolerance for nll increase resulted from numerical design around optimization constraints
    num_rand_init_phi_update = 2  # rand init to avoid local minimum in clus assignment beyond init by current estimate
    num_max_iter = 20
    eta_for_phi = eta/10.

    def __init__(self, w, pair_to_node, model_file, num_clus, beta=1.e-4):
        assert sp.isspmatrix_dok(w), "Input w must be a scipy DOK-based sparse matrix."
        self.w = w.asfptype()  # sparse matrix, it must consist of float numbers
        self.pair_to_node = pair_to_node  # [(, )]
        assert self.sym_checker(), "Node pairs duplicated in input matrix."
        self.node_to_pair = self.find_node_to_pair()  # [[(, )]]
        self.model_file = model_file

        self.num_pair = self.w.shape[0]  # S
        self.num_type = self.w.shape[1]  # T
        self.num_node = len(self.node_to_pair)  # V
        self.num_clus = num_clus  # K
        assert self.num_clus >= self.num_type, "Number of clusters must be greater than or equal to number of meta-paths."
        self.beta = beta
        self.alpha = self.num_type * (self.num_node - 1.) + 1.

    def pre_fit_initialize(self):
        self.mu = np.ones(self.num_type)

        self.rho = np.random.gamma(self.alpha, size=self.num_node)

        self.phi = self.gen_rand(self.num_pair, self.num_clus)

        self.theta = np.vstack((
            np.eye(self.num_type),
            self.gen_rand(self.num_clus - self.num_type, self.num_type)
        ))
        self.theta = (self.theta.transpose() * ((1. - self.num_clus * self.delta_D) / np.sum(self.theta, axis=1).transpose())).transpose() + self.delta_D

    def precompute_composite_parameters(self):
        self.update_tau()
        self.update_phi_times_theta()
        self.w_row_nnz = np.asarray([self.w[s, :].nnz for s in xrange(self.num_pair)])
        self.w_col_nnz = np.asarray([self.w[:, t].nnz for t in xrange(self.num_type)])

    def fit(self, max_iter=100):
        self.pre_fit_initialize()
        self.precompute_composite_parameters()

        iter_num = 0
        nll = self.neg_log_likelihood()
        diff_ratio = 1.
        ts = datetime.datetime.now()

        np.set_printoptions(precision=2, suppress=True)

        while iter_num < max_iter and abs(diff_ratio) > self.eta:
            np.set_printoptions(precision=2, suppress=True)

            self.update_mu()
            print "mu updated.    ", "time_passed:", datetime.datetime.now() - ts
            if np.any(np.isnan(self.mu)):
                print "mu NaN!", self.mu
                break

            self.update_rho()
            print "rho updated.   ", "time_passed:", datetime.datetime.now() - ts
            if np.any(np.isnan(self.rho)):
                print "rho NaN!", self.rho
                break

            self.update_phi(eta_for_phi=self.eta_for_phi)
            print "phi updated.   ", "time_passed:", datetime.datetime.now() - ts
            if np.any(np.isnan(self.phi)):
                print "phi NaN!", self.phi
                break

            self.update_theta()
            print "theta updated. ", "time_passed:", datetime.datetime.now() - ts
            if np.any(np.isnan(self.theta)):
                print "theta NaN!", self.theta
                break

            nll_new = self.neg_log_likelihood()
            # stop criterion
            print "iter:", iter_num, "time_passed:", datetime.datetime.now() - ts, "nll:", nll_new
            sys.stdout.flush()
            self.save_model()
            print "Model saved!"
            sys.stdout.flush()

            iter_num += 1
            diff_ratio = (nll_new - nll)/abs(nll)
            nll = nll_new
        print "*** Model fitting done. ***"

    def save_model(self):
        model_file = file(self.model_file, "wb")
        np.save(model_file, self.num_clus)
        np.save(model_file, self.beta)
        np.save(model_file, self.alpha)
        np.save(model_file, self.phi)
        np.save(model_file, self.theta)
        np.save(model_file, self.mu)
        np.save(model_file, self.rho)
        model_file.close()

    def load_model(self):
        model_file = file(self.model_file, "rb")
        self.num_clus = int(np.load(model_file))
        self.beta = float(np.load(model_file))
        self.alpha = float(np.load(model_file))
        self.phi = np.load(model_file)
        self.theta = np.load(model_file)
        self.mu = np.load(model_file)
        self.rho = np.load(model_file)
        model_file.close()
        self.precompute_composite_parameters()

    def sym_checker(self):
        pairs = set()
        for pair in self.pair_to_node:
            if pair[::-1] in pairs or pair in pairs:
                return False
            pairs.add(pair)
        return True

    def find_node_to_pair(self):
        node_to_pair = [[] for _ in range(max(map(max, *self.pair_to_node)) + 1)]
        for s, (u, v) in enumerate(self.pair_to_node):
            node_to_pair[u].append((v, s))
            node_to_pair[v].append((u, s))
        return node_to_pair

    @staticmethod
    def gen_rand(num_x, num_y):
        mat = np.random.rand(num_x, num_y)
        return mat / mat.sum(axis=1)[:, None]

    def update_tau(self):
        self.tau = np.asarray(map(lambda (u, v): self.rho[u] * self.rho[v], self.pair_to_node))

    def update_phi_times_theta(self):
        self.phi_times_theta = self.phi.dot(self.theta)

    @staticmethod
    def log_beta_func(param, rep):
        return rep * np.log(gamma(param)) - np.log(gamma(rep * param))

    def neg_log_likelihood(self):
        t1 = np.sum(self.rho)
        t2 = - (self.alpha-1.) * np.sum(np.log(self.rho))
        t3 = - (self.beta-1.) * np.sum(delta_D_bounded_log(self.phi))
        t4 = self.num_type * (self.num_node - 1) * np.sum(np.log(self.rho))
        t5 = self.w_col_nnz.dot(np.log(self.mu))
        t6 = np.sum(delta_E_bounded_log(self.phi_times_theta))
        mat_prod = ((self.phi_times_theta * self.mu).transpose() * self.tau).transpose()
        t7 = sum([value / mat_prod[key] for (key, value) in self.w.items()])
        return t1 + t2 + t3 + t4 + t5 + t6 + t7

    def conditional_neg_log_likelihood(self):
        t1_1 = self.w_row_nnz.dot(np.log(self.tau))
        t1_2 = self.num_type * np.sum(np.log(self.tau))
        t2 = self.w_col_nnz.dot(np.log(self.mu))
        t3 = np.sum(delta_E_bounded_log(self.phi_times_theta))
        mat_prod = ((self.phi_times_theta * self.mu).transpose() * self.tau).transpose()
        t4 = sum([value / mat_prod[key] for (key, value) in self.w.items()])
        return t1_1 + t2 + t3 + t4, t1_2 + t2 + t3 + t4

    def pair_neg_log_likelihood(self, s):
        mat_prod__s = self.phi_times_theta[s,:] * self.mu * self.tau[s]
        t1 = sum([value / mat_prod__s[_t] for ((_s, _t), value) in self.w.items() if _s == s])
        t2 = np.sum(delta_E_bounded_log(mat_prod__s))
        t3 = - (self.beta-1.) * np.sum(delta_D_bounded_log(self.phi[s,:]))
        return t1 + t2 + t3

    def all_pairs_neg_log_likelihood(self):
        mat_prod = ((self.phi_times_theta * self.mu).transpose() * self.tau).transpose()
        t1 = np.zeros(self.num_pair)
        for ((s, t), value) in self.w.items():
            t1[s] += value / mat_prod[s, t]
        t2 = np.sum(delta_E_bounded_log(mat_prod), axis=1)
        t3 = - (self.beta-1.) * np.sum(delta_D_bounded_log(self.phi), axis=1)
        return t1 + t2 + t3

    def new_all_pairs_neg_log_likelihood(self):
        mat_prod = ((self.phi_times_theta * self.mu).transpose() * self.tau).transpose()
        t1 = np.zeros(self.num_pair)
        for ((s, t), value) in self.w.items():
            t1[s] += value / mat_prod[s, t]
        t3 = - (self.beta-1.) * np.sum(delta_D_bounded_log(self.phi), axis=1)
        return t1 + t3

    def update_mu(self):
        self.mu = np.zeros(self.num_type)
        for ((s, t), value) in self.w.items():
            self.mu[t] += value / self.tau[s] / self.phi_times_theta[(s, t)]
        self.mu = 1.*self.mu / self.w_col_nnz

    def update_rho(self):
        nll = self.neg_log_likelihood()
        diff_ratio = 1.0
        xi = np.zeros(self.num_pair)
        for ((s, t), value) in self.w.items():
            xi[s] += value / self.mu[t] / self.phi_times_theta[(s, t)]
        num_iter = 0
        while abs(diff_ratio) > self.eta:
            for u, node_to_pair_u in enumerate(self.node_to_pair):  # pairs: [(v, s)]
                b = self.num_type * (self.num_node - 1) - (self.alpha - 1.)
                c = - np.sum(map(lambda v_s: xi[v_s[1]] / self.rho[v_s[0]], node_to_pair_u))
                self.rho[u] = (-b + np.sqrt(b**2. - 4.*c)) / 2.
            self.rho[np.where(self.rho < 1.)] = 1.
            # update tau before recomputing nll
            self.update_tau()
            nll_new = self.neg_log_likelihood()
            diff_ratio = (nll_new - nll) / abs(nll)
            nll = nll_new
            num_iter += 1
            if num_iter > self.num_max_iter:
                print "rho update does not converge."
                break

    # update sigma as a whole to ensure convergence
    # DEPRECATED -- model for lambda has been changed
    @deprecated
    def update_sigma(self):
        # \sum_t (w_{s, t} / (mu_t * (phi*theta)_{s, t}))
        xi = np.zeros(self.num_pair)
        for ((s, t), value) in self.w.items():
            xi[s] += value / self.mu[t] / self.phi_times_theta[(s, t)]

        # objective function w.r.t. sigma
        def cur_obj(x): return self.update_sigma_obj(x, self.pair_to_node, xi, self.num_type)
        # Jacobian w.r.t. sigma
        def cur_jac(x): return self.update_sigma_jac(x, self.node_to_pair, xi, self.num_type, self.num_node)

        # bounds, constraints, and initial value (converted to 1-D array)
        cur_bnds = tuple([(self.sigma_lower, None) for _ in xrange(self.num_node)])
        sigma_init = self.sigma

        # optimize
        opt = minimize(cur_obj, sigma_init, method='L-BFGS-B', jac=cur_jac, bounds=cur_bnds)

        # update sigma and lambda
        self.sigma = opt.x
        self.update_tau()

    @staticmethod
    @deprecated
    def update_sigma_obj(sigma, pair_to_node, xi, num_type):
        cur_tau = np.asarray(map(lambda (u, v): 2./(1./sigma[u] + 1./sigma[v]), pair_to_node))
        t1 = num_type * np.sum(np.log( cur_tau ))
        t2 = (1./cur_tau).dot(xi)
        return t1 + t2

    @staticmethod
    @deprecated
    def update_sigma_jac(sigma, node_to_pair, xi, num_type, num_node):
        jac = np.zeros(num_node)
        for u in xrange(num_node):
            sigma__u = sigma[u]
            assert isinstance(sigma__u, float), "sigma__u is not float, may arouse computational error in division."
            sigma__v_arr = np.asarray(map(lambda v_s_tuple: sigma[v_s_tuple[0]], node_to_pair[u]))
            xi_arr = np.asarray(map(lambda v_s_tuple: xi[v_s_tuple[1]], node_to_pair[u]))
            tau_arr = 2./(1./sigma__u + 1./sigma__v_arr)
            jac[u] = (2. / (sigma__u/sigma__v_arr + 1.)**2).dot(num_type/tau_arr - 1./tau_arr**2 * xi_arr)

        return jac

    # DEPRECATED -- jointly update sigma as a whole to ensure convergence
    @deprecated
    def update_sigma_element_wise(self):
        nll = self.neg_log_likelihood()
        diff_ratio = 1.0
        xi = np.zeros(self.num_pair) # \sum_t (w_{s, t} / (mu_t * (phi*theta)_{s, t}))
        for ((s, t), value) in self.w.items():
            xi[s] += value / self.mu[t] / self.phi_times_theta[(s, t)]
        num_iter = 0
        while abs(diff_ratio) > self.eta:
            sigma_updated = Parallel(n_jobs=multiprocessing.cpu_count())\
                (delayed(update_sigma_one_element_wrapper)
                 (u, self.node_to_pair[u], xi, self.num_type, self.num_node, self.sigma)
                 for u in xrange(self.num_node))  # a list of float is returned

            # copy back to sigma and update tau before recomputing nll
            self.sigma = np.asarray(sigma_updated)
            self.update_tau()
            nll_new = self.neg_log_likelihood()
            print nll, nll_new
            diff_ratio = (nll_new - nll) / abs(nll)
            nll = nll_new
            num_iter += 1
            if num_iter > self.num_max_iter:
                print "sigma update does not converge."
                break

    # DEPRECATED -- jointly update sigma as a whole to ensure convergence
    @staticmethod
    @deprecated
    def update_sigma_one_element_obj(sigma__u, sigma__v_arr, xi_arr, num_type):
        t1 = num_type * np.sum(np.log( 2./(1./sigma__u + 1./sigma__v_arr) ))
        t2 = ((1./sigma__u + 1./sigma__v_arr)/2.).dot(xi_arr)
        return t1 + t2

    # different from theta, update row by row for having too many rows and being decomposible
    def update_phi(self, eta_for_phi):
        # compute w_over_tau_mu
        w_over_tau_mu = sp.dok_matrix(self.w)
        for ((s, t), value) in w_over_tau_mu.items():
            w_over_tau_mu[(s, t)] = 1.*self.w[(s, t)] / (self.tau[s] * self.mu[t])
        w_over_tau_mu = w_over_tau_mu.toarray()

        # update row by row with multiprocessing, each process with block of rows
        # Note: workaround by invoking wrapper function outside of current class because pickle in joblib do not
        #       support pickling instancemethod
        # block_num = min(10, self.num_pair)
        block_num = min(multiprocessing.cpu_count(), self.num_pair)
        block_size = int(np.ceil(float(self.num_pair) / block_num))
        block_ranges = [(i, min(i+block_size, self.num_pair)) for i in xrange(0, self.num_pair, block_size)]

        phi_opt = Parallel(n_jobs=block_num)\
            (delayed(update_phi_one_row_wrapper)
             (w_over_tau_mu[block_ranges[i][0]:block_ranges[i][1], :], self.phi[block_ranges[i][0]:block_ranges[i][1], :],
              self.theta, self.beta, self.num_clus, eta_for_phi, self.delta_D)
             for i in xrange(block_num))

        # copy back to phi
        self.phi = np.vstack(tuple(phi_opt))
        # update phi*theta once
        self.update_phi_times_theta()

    # all vectors are 2-D array (1, num_clus)
    @staticmethod
    def update_phi_one_row_obj(phi__s, theta, w_over_tau_mu__s, beta):
        t1 = - (beta-1.) * np.sum(delta_D_bounded_log(phi__s))
        cur_phi_times_theta__s = phi__s.dot(theta)
        t2 = np.sum(delta_E_bounded_log(cur_phi_times_theta__s))
        t3 = np.sum(w_over_tau_mu__s / cur_phi_times_theta__s)
        return t1 + t2 + t3

    # all vectors are 2-D array (1, num_clus)
    @staticmethod
    def update_phi_one_row_jac(phi__s, theta, w_over_tau_mu__s, beta):
        t1 = - delta_D_bounded_divide(beta-1., phi__s)
        cur_phi_times_theta__s = phi__s.dot(theta)
        t2 = (delta_E_bounded_divide(1., cur_phi_times_theta__s) - w_over_tau_mu__s/(cur_phi_times_theta__s**2)).dot(theta.transpose())
        return t1 + t2

    def update_theta(self):
        # compute w_over_tau_mu
        w_over_tau_mu = sp.dok_matrix(self.w)
        for ((s, t), value) in w_over_tau_mu.items():
            w_over_tau_mu[(s, t)] = 1.*self.w[(s, t)] / (self.tau[s] * self.mu[t])
        w_over_tau_mu = w_over_tau_mu.toarray()

        # objective function w.r.t. theta
        def cur_obj(x): return self.update_theta_obj(x, self.phi, w_over_tau_mu)

        # Jacobian w.r.t. theta
        def cur_jac(x): return self.update_theta_jac(x, self.phi, w_over_tau_mu)

        # optimize
        self.theta, _ = gradient_descent(self.theta, cur_obj, cur_jac, self.eta, self.delta_D*2., step_limit=1000, step_len_init=0.00001)

        # update phi*theta once
        self.update_phi_times_theta()

    @staticmethod
    def update_theta_obj(theta, phi, w_over_tau_mu):
        cur_phi_times_theta = phi.dot(theta)
        t1 = np.sum(delta_E_bounded_log(cur_phi_times_theta))
        t2 = np.sum(w_over_tau_mu / cur_phi_times_theta)
        return t1 + t2

    @staticmethod
    def update_theta_jac(theta, phi, w_over_tau_mu):
        cur_phi_times_theta = phi.dot(theta)
        return (phi.transpose()).dot(delta_E_bounded_divide(1., cur_phi_times_theta) - w_over_tau_mu / (cur_phi_times_theta ** 2))

    # DEPRECATED
    @deprecated
    def update_theta_parallel(self):
        # compute w_over_tau_mu; used full matrix for package compatibility
        w_over_tau_mu = np.zeros(self.w._shape)
        for ((s, t), value) in self.w.items():
            w_over_tau_mu[s, t] = 1.*self.w[(s, t)] / (self.tau[s] * self.mu[t])

        # update row by row with multiprocessing for iterations
        # though each row is not independent, we sacrifice as such to ensure optimization of each single row
        nll = self.neg_log_likelihood()
        diff_ratio = 1.0
        num_iter = 0
        parallel_theta_update = True
        ts = datetime.datetime.now()
        while abs(diff_ratio) > self.eta:
            if parallel_theta_update is True:
                # Note: unexpected behavior exists when any single row cannot be further updated. In such case,
                # no row in any thread will be updated.
                theta_old = np.copy(self.theta)
                theta_updated = Parallel(n_jobs=multiprocessing.cpu_count()) \
                    (delayed(update_theta_one_row_wrapper)
                     (k, w_over_tau_mu,
                      self.phi, self.num_type, self.theta, self.delta_D)
                     for k in xrange(self.num_clus))

                # copy back to theta and update phi*theta
                self.theta = np.asarray(theta_updated)
                self.update_phi_times_theta()
            else:
                clus_shuffled = range(self.num_clus)
                shuffle(clus_shuffled)
                for k in clus_shuffled:
                    self.theta[k] = update_theta_one_row_wrapper(k, w_over_tau_mu, self.phi,
                                                                 self.num_type, self.theta, self.delta_D)

                self.update_phi_times_theta()

            nll_new = self.neg_log_likelihood()
            print "theta: iter", num_iter, "diff:", (nll_new - nll) / abs(nll), "time_passed_in_update_theta:", datetime.datetime.now() - ts
            if nll_new - nll > self.nll_inc_tol:
                print "Significant nll increase encountered in theta update iter "+str(num_iter)+". Revert and trigger one-time nonparallel theta update."
                if parallel_theta_update is False:
                    print "Nonparallel theta update used, but nll still significantly increased."
                self.theta = theta_old
                self.update_phi_times_theta()
                parallel_theta_update = False
            else:
                diff_ratio = (nll_new - nll) / abs(nll)
                nll = nll_new
                num_iter += 1
                parallel_theta_update = True  # parallel theta update for the next iter

            if num_iter > self.num_max_iter:
                print "theta update does not converge."
                break

    # DEPRECATED
    @staticmethod
    @deprecated
    def update_theta_one_row_obj(theta__k, phi, w_over_tau_mu, theta_old, k):
        # substitute current row for theta will not update until end of multiprocess
        theta = np.copy(theta_old)
        theta[k] = theta__k

        cur_phi_times_theta = phi.dot(theta)
        t1 = np.sum(delta_E_bounded_log(cur_phi_times_theta))
        t2 = np.sum(w_over_tau_mu / cur_phi_times_theta)
        return t1 + t2

    # DEPRECATED
    @staticmethod
    @deprecated
    def update_theta_one_row_jac(theta__k, phi, w_over_tau_mu, theta_old, k):
        # substitute current row for theta will not update until end of multiprocess
        theta = np.copy(theta_old)
        theta[k] = theta__k

        cur_phi_times_theta = phi.dot(theta)
        return (phi[:, [k]].transpose()).dot(delta_E_bounded_divide(1., cur_phi_times_theta) - w_over_tau_mu / (cur_phi_times_theta**2))

    def top_n_node(self, u, n):  # u: query node, n: top n
        node_score_dist = [(v, self.pair_neg_log_likelihood(s), self.phi[s, :]) for (v, s) in self.node_to_pair[u]]
        return sorted(node_score_dist, key=lambda x: x[1], reverse=True)[:n]


# Outside of PReP since parallelization package joblib does not support instance method
# DEPRECATED -- jointly update sigma as a whole to ensure convergence
@deprecated
def update_sigma_one_element_wrapper(u, node_to_pair_u, xi, num_type, num_node, sigma_old):
    # compute quantities involving pairs having u
    sigma__v_arr = np.asarray(map(lambda v_s: sigma_old[v_s[0]], node_to_pair_u))
    xi_arr = np.asarray(map(lambda v_s: xi[v_s[1]], node_to_pair_u))

    # objective function w.r.t. sigma__u
    def cur_obj(x): return PReP.update_sigma_one_element_obj(x, sigma__v_arr, xi_arr, num_type)

    # bounds
    bnds = (0., 1e10)

    # solve optimization and return
    opt = minimize_scalar(cur_obj, bounds=bnds, method="bounded")
    return opt.x


# Outside of PReP since parallelization package joblib does not support instance method
def update_phi_one_row_wrapper(w_over_tau_mu__block, phi__block, theta, beta, num_clus, eta, delta_D):
    block_min_opt_arg_list = []
    block_size = w_over_tau_mu__block.shape[0]
    # print "block_size:", block_size
    for s in xrange(block_size):
        w_over_tau_mu__s = w_over_tau_mu__block[s, :]

        # objective function w.r.t. phi__s (input: 2-D array)
        def cur_obj(x): return PReP.update_phi_one_row_obj(x, theta, w_over_tau_mu__s, beta)

        # Jacobian w.r.t. phi__s (input and output: 2-D array)
        def cur_jac(x): return PReP.update_phi_one_row_jac(x, theta, w_over_tau_mu__s, beta)

        # initial value
        phi__s_old = np.resize(phi__block[s, :], (1, num_clus))  # 2-D np array

        opt_arg_list = []
        opt_val_list = []

        # solve optimization with current estimate as initialization
        phi__s_init = phi__s_old
        # opt = minimize(cur_obj, phi__s_init, method='SLSQP', jac=cur_jac, bounds=cur_bnds, constraints=cur_cons)
        opt_arg, opt_val = gradient_descent(phi__s_init, cur_obj, cur_jac, eta, delta_D*2., step_limit=1000)
        opt_arg_list.append(opt_arg)
        opt_val_list.append(opt_val)

        # solve optimization with random initialization
        for i in xrange(PReP.num_rand_init_phi_update):
            phi__s_init = PReP.gen_rand(1, num_clus)
            opt_arg, opt_val = gradient_descent(phi__s_init, cur_obj, cur_jac, eta, delta_D*2., step_limit=1000)
            opt_arg_list.append(opt_arg)
            opt_val_list.append(opt_val)

        # find best opt
        min_opt_idx = min(xrange(len(opt_val_list)), key=opt_val_list.__getitem__)
        min_opt_arg = opt_arg_list[min_opt_idx]

        if cur_obj(min_opt_arg) - cur_obj(phi__s_old) > PReP.nll_inc_tol:
            print "Significant nll increase encountered in updating phi:", cur_obj(min_opt_arg) - cur_obj(phi__s_old)

        block_min_opt_arg_list.append(np.resize(min_opt_arg, num_clus))
    return np.asarray(block_min_opt_arg_list)


# Outside of PReP since parallelization package joblib does not support instance method
# DEPRECATED
@deprecated
def update_theta_one_row_wrapper(k, w_over_tau_mu, phi, num_type, theta_old, delta_D):
    # objective function w.r.t. theta__k
    def cur_obj(x): return PReP.update_theta_one_row_obj(x, phi, w_over_tau_mu, theta_old, k)

    # Jacobian w.r.t. phi__s
    def cur_jac(x): return PReP.update_theta_one_row_jac(x, phi, w_over_tau_mu, theta_old, k)

    # bounds, constraints, and initial value
    cur_bnds = tuple([(delta_D*2., 1.-delta_D*2.) for _ in xrange(num_type)])
    cur_cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1, 'jac': lambda x: np.ones(num_type)})
    theta__k_init = theta_old[k, :]  # 1-D np array

    # solve optimization and return
    opt = minimize(cur_obj, theta__k_init, method='SLSQP', jac=cur_jac, bounds=cur_bnds, constraints=cur_cons)
    return opt.x


## two delta bounded functions for Dirichlet
# x as numpy array.
# Outside of PReP for being called by function called by update_phi_one_row_wrapper
def delta_D_bounded_log(x):
    log_delta = np.log(PReP.delta_D)
    bounded_log_x = np.ones_like(x, dtype=np.float) * log_delta
    bounded_log_x[x > PReP.delta_D] = np.log(x[x > PReP.delta_D])
    return bounded_log_x

# a as float, x as numpy array; a / x
# Outside of PReP for being called by function called by update_phi_one_row_wrapper
# NOTE: this function should only be used for division resulted from taking derivative of log
def delta_D_bounded_divide(a, x):
    bounded_a_over_x = np.zeros_like(x, dtype=np.float)
    bounded_a_over_x[x > PReP.delta_D] = a / (x[x > PReP.delta_D])
    return bounded_a_over_x



## two delta bounded functions for Exponential
# x as numpy array.
# Outside of PReP for being called by function called by update_phi_one_row_wrapper
def delta_E_bounded_log(x):
    log_delta = np.log(PReP.delta_E)
    bounded_log_x = np.ones_like(x, dtype=np.float) * log_delta
    bounded_log_x[x > PReP.delta_E] = np.log(x[x > PReP.delta_E])
    return bounded_log_x

# a as float, x as numpy array; a / x
# Outside of PReP for being called by function called by update_phi_one_row_wrapper
# NOTE: this function should only be used for division resulted from taking derivative of log
def delta_E_bounded_divide(a, x):
    bounded_a_over_x = np.zeros_like(x, dtype=np.float)
    bounded_a_over_x[x > PReP.delta_E] = a / (x[x > PReP.delta_E])
    return bounded_a_over_x
