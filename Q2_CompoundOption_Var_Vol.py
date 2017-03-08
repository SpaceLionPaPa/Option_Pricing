'''This is second program for term project of Derivatives modeling. And this program is aim to price
at_the_money compound options with time-varying volatility'''
'Run successfully in 3.0 version'
import numpy as np
from numpy import exp, sqrt, maximum, mean, cov, var, std, floor, array
from numpy.random import randn, seed
from BS import bsformula
from scipy.stats import mvn, norm
import scipy.linalg as linalg

# Create a class of compound options whose volatility is time-varying
__mataclass__=type # Use new type classes
class Comp_Opt_Var_Vol:
    def __init__(self, S0, X1, X2, r, T1, T2, mu1,mu2, sigma1, sigma2):
        self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.mu1, self.mu2, self.sigma1, self.sigma2 =\
            [S0, X1, X2, r, T1, T2, mu1, mu2, sigma1, sigma2]

    def simple_Monte_Carlo(self, underlying, N=10000, sd=None):
        #obtain local augs from public augs
        S0, X1, X2, r, T1, T2, sigma1, sigma2 = \
            [self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.sigma1, self.sigma2]

        # simulate prices
        seed(sd)  # set seed for simulator
        nu = r - sigma1 ** 2 / 2.0; nuT1 = nu * T1; sigmasqrtT1 = sigma1 * sqrt(T1)
        ST1 = S0 * exp(nuT1 + sigmasqrtT1 * randn(1, N))

        # calculate compound options' prices using simulated underlying assets' prices at T1
        if underlying == 'Call':  # When underlying option is Call
            c = bsformula(1, ST1, X2, r, T2 - T1, sigma2)[0]
            cc, pc = [exp(-r * T1) * maximum(c - X1[0], 0.0), exp(-r * T1) * maximum(X1[0] - c, 0.0)]  # Compound option's price at T1
            call_On_Call, put_On_Call = mean(cc), mean(pc)

            # Confidential interval
            loc1, scale1 = norm.fit(cc)
            CI = norm.interval(0.05, loc=loc1, scale=scale1)
            return [call_On_Call, put_On_Call, CI, cc]
        if underlying == 'Put':  # When underlying option is Put
            p = bsformula(-1, ST1, X2, r, T2 - T1, sigma2)[0]
            cp, pp = [exp(-r * T1) * maximum(p - X1[1], 0.0), exp(-r * T1) * maximum(X1[1] - p, 0.0)]
            call_On_Put, put_On_Put = mean(cp), mean(pp)

            # Confidential interval
            loc2, scale2 = norm.fit(cp)
            CI = norm.interval(0.05, loc=loc2, scale=scale2)
            return [call_On_Put, put_On_Put, CI, cp]

    def smart_Lattices_CRR_Bi_Tree(self, N):
        # obtain local augs from public augs
        S0, X1, X2, r, T1, T2, sigma1, sigma2 = \
            [self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.sigma1, self.sigma2]
        # set basic augs
        deltaT = T1 / N
        u = exp(sigma1 * sqrt(deltaT)); d = 1. / u
        p = (exp(r * deltaT) - d) / (u - d)
        discount = exp(-r * deltaT)
        p_u = discount * p; p_d = discount * (1 - p)

        # build grids for lattices
        SVals, CVals, PVals, CCVals, PCVals, CPVals, PPVals = \
            [np.zeros((2 * N + 1, 1)) for _ in range(7)]
        SVals[0] = S0 * d ** N
        CVals[0] = bsformula(1, SVals[0], X2, r, T2 - T1, sigma2)[0]
        PVals[0] = bsformula(-1, SVals[0], X2, r, T2 - T1, sigma2)[0]
        for i in range(1, 2 * N + 1):  # prices in T
            SVals[i] = u * SVals[i - 1]
            CVals[i] = bsformula(1, SVals[i], X2, r, T2 - T1, sigma2)[0]
            PVals[i] = bsformula(-1, SVals[i], X2, r, T2 - T1, sigma2)[0]
        for i in range(0, 2 * N + 1, 2):  # price of compound options at T
            CCVals[i] = max(CVals[i] - X1[0], 0.0)
            PCVals[i] = max(X1[0] - CVals[i], 0.0)
            CPVals[i] = max(PVals[i] - X1[1], 0.0)
            PPVals[i] = max(X1[1] - PVals[i], 0.0)

        # backward using smart lattices
        for tau in range(1, N + 1):
            for i in range(tau, 2 * N + 1 - tau, 2):
                CCVals[i] = p_u * CCVals[i + 1] + p_d * CCVals[i - 1]
                PCVals[i] = p_u * PCVals[i + 1] + p_d * PCVals[i - 1]
                CPVals[i] = p_u * CPVals[i + 1] + p_d * CPVals[i - 1]
                PPVals[i] = p_u * PPVals[i + 1] + p_d * PPVals[i - 1]
        price = [CCVals[N], PCVals[N], CPVals[N], PPVals[N]]
        return price

    def implicit_Finite_Diff(self, Smax, dS, dt):
        # obtain local augs from public augs
        S0, X1, X2, r, T1, T2, sigma1, sigma2 = \
            [self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.sigma1, self.sigma2]
        # set basic augs
        M = int(round(Smax / dS)); ds = Smax/ M; N = int(round(T1 / dt)); dt = T1 / N
        # set grids from T0 to T1
        matvalCC, matvalPC, matvalCP, matvalPP = [np.zeros((M + 1, N + 1)) for _ in range(4)]
        vetS = np.linspace(int(0), Smax, M + 1)
        veti = np.arange(0, M + 1); vetj = np.arange(0, N + 1)
        vetC = bsformula(1, vetS, X2, r, T2 - T1, sigma2)[0]; vetP = bsformula(-1, vetS, X2, r, T2 - T1, sigma2)[0]

        # last column at T1 for different compound options
        matvalCC[:, -1] = maximum(vetC - X1[0], 0.0)
        matvalPC[:, -1] = maximum(X1[0]- vetC, 0.0)
        matvalCP[:, -1] = maximum(vetP - X1[1], 0.0)
        matvalPP[:, -1] = maximum(X1[1]- vetP, 0.0)
        # Upper boundary and Lower bound for different compound options
        matvalCC[0, :] = 0.0 # Upper boundary for Call on Call option
        matvalCC[-1, :] = max(bsformula(1, Smax, X2, r, T2-T1, sigma2)[0] - X1[0], 00) * exp(-r*dt*(N-vetj)) # Lower boundary for Call on Call option
        matvalPC[0, :] = X1[0] * exp(-r*dt*(N-vetj))
        matvalPC[-1, :] = max(X1[0] - bsformula(1, Smax, X2, r, T2-T1, sigma2)[0], 0.0) * exp(-r*dt*(N-vetj))
        matvalCP[0, :] = max(bsformula(-1, 0.0, X2, r, T2-T1, sigma2)[0] - X1[1], 0.0) * exp(-r*dt*(N-vetj))
        matvalCP[-1, :] = max(bsformula(-1, Smax, X2, r, T2-T1, sigma2)[0] - X1[1], 0.0) * exp(-r*dt*(N-vetj))
        matvalPP[0, :] = max(X1[1]- bsformula(-1, 0.0, X2, r, T2-T1, sigma2)[0], 0.0) * exp(-r*dt*(N-vetj))
        matvalPP[-1, :] = max(X1[1]- bsformula(-1, Smax, X2, r, T2-T1, sigma2)[0], 0.0) * exp(-r*dt*(N-vetj))

        # Set grid coefficients
        a = 0.5*(r * dt * veti - (sigma1 ** 2) * dt * (veti**2))
        b = 1 + (sigma1 ** 2) * dt * (veti ** 2) + r * dt
        c = (-0.5) * (r * dt * veti + (sigma1 ** 2) * dt * (veti ** 2))
        coeff = np.diag(a[2: M], -1) + np.diag(b[1: M]) + np.diag(c[1: M-1], +1)

        # Traverse_grid
        matval = [matvalCC, matvalPC, matvalCP, matvalPP]; price = [0] * 4; num = 0
        for w in matval:
            """Solving using linear system of equations"""
            P, L, U = linalg.lu(coeff)
            # print("shape of L and U", np.shape(L), np.shape(U))
            aux = np.zeros(M - 1)
            for j in reversed(range(N)):  # Traverse grid backwardly
                aux[0] = np.dot(-a[1], w[0, j]); aux[M - 2] = np.dot(-c[M - 1], w[M, j])
                y = linalg.solve(L, (w[1:M, j + 1] + aux))
                x = linalg.solve(U, y)  # 1 to M-1, the M row is the stop point
                w[1:M, j] = x  # 1 to M-1, the M row is the stop point
            # interpolate result
            price[num] = np.interp(S0, vetS, w[:, 0])
            num += 1
        return price

    '''Variance Reduction'''
    def antiSampling(self, underlying, N=10000, sd=None):
        # obtain local augs from public augs
        S0, X1, X2, r, T1, T2, sigma1, sigma2 = \
            [self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.sigma1, self.sigma2]

        # simulate prices
        seed(sd)  # set seed for simulator
        nu = r - sigma1 ** 2 / 2.0; nuT1 = nu * T1; sigmasqrtT1 = sigma1 * sqrt(T1)
        a1= randn(1, N)
        ST11 = S0 * exp(nuT1 + sigmasqrtT1 * a1); ST12 = S0 * exp(nuT1 + sigmasqrtT1 * (-a1))

        # calculate compound options' prices using simulated underlying assets' prices at T1
        if underlying == 'Call':  # When underlying option is Call
            c1 = bsformula(1, ST11, X2, r, T2 - T1, sigma2)[0]
            c2 = bsformula(1, ST12, X2, r, T2 - T1, sigma2)[0]

            cc1, pc1 = [exp(-r * T1) * maximum(c1 - X1[0], 0.0),
                      exp(-r * T1) * maximum(X1[0] - c1, 0.0)]  # Compound option's price at T1
            cc2, pc2 = [exp(-r * T1) * maximum(c2 - X1[0], 0.0),
                      exp(-r * T1) * maximum(X1[0] - c2, 0.0)]  # Compound option's price at T1

            cc = (cc1 + cc2) / int(2); pc = (pc1 + pc2) / int(2)  # Average the pair
            call_On_Call, put_On_Call = mean(cc), mean(pc)

            # Confidential interval
            loc1, scale1 = norm.fit(cc)
            CI = norm.interval(0.05, loc=loc1, scale=scale1)
            return [call_On_Call, put_On_Call, CI, cc]
        if underlying == 'Put':  # When underlying option is Put
            p1 = bsformula(-1, ST11, X2, r, T2 - T1, sigma2)[0]
            p2 = bsformula(-1, ST12, X2, r, T2 - T1, sigma2)[0]

            cp1, pp1 = [exp(-r * T1) * maximum(p1 - X1[1], 0.0),
                        exp(-r * T1) * maximum(X1[1] - p1, 0.0)]
            cp2, pp2 = [exp(-r * T1) * maximum(p2 - X1[1], 0.0),
                        exp(-r * T1) * maximum(X1[1] - p2, 0.0)]

            cp = (cp1 + cp2) / int(2); pp = (pp1 + pp2) / int(2)
            call_On_Put, put_On_Put = mean(cp), mean(pp)

            # Confidential interval
            loc2, scale2 = norm.fit(cp)
            CI = norm.interval(0.05, loc=loc2, scale=scale2)
            return [call_On_Put, put_On_Put, CI, cp]

    def controlVariates(self, underlying, N= int(10000), NTrials = int(10000), sd=None):
        # obtain local augs from public augs
        S0, X1, X2, r, T1, T2, sigma1, sigma2 = \
            [self.S0, self.X1, self.X2, self.r, self.T1, self.T2, self.sigma1, self.sigma2]

        # simulate prices
        seed(sd)  # set seed for simulator
        nu = r - sigma1 ** 2 / 2.0; nuT1 = nu * T1; sigmasqrtT1 = sigma1 * sqrt(T1)

        # first, use NTrials to get optimal coefficient b* , NTrials value can be not too large
        ST1 = S0 * exp(nuT1 + sigmasqrtT1 * randn(1, NTrials))
        # calculate compound options' prices using simulated underlying assets' prices at T1
        if underlying == 'Call':  # When underlying option is Call
            Euc = maximum(ST1 - X2, 0.) # x pair
            c = bsformula(1, ST1, X2, r, T2 - T1, sigma2)[0]
            cc, pc = [exp(-r * T1) * maximum(c - X1[0], 0.0), # Y pair
                      exp(-r * T1) * maximum(X1[0] - c, 0.0)]  # Compound option's price at T1
            VarCov1 = cov(Euc, cc); c_coeff1= VarCov1[0,1]/var(Euc)
            VarCov2 = cov(Euc, pc); c_coeff2 = VarCov2[0, 1] / var(Euc)

            # use N to get the real simulated data
            ST1 = S0 * exp(nuT1 + sigmasqrtT1 * randn(1, N))
            Euc = maximum(ST1 - X2, 0.)
            '''calculate the expected value for Euc using Black-Scholes formula'''
            EEuc = exp(r*T1)*bsformula(1, S0, X2, r, T2 - T1, sigma1)[0]
            c = bsformula(1, ST1, X2, r, T2 - T1, sigma2)[0]
            # control variate --c_coeff1 * (Euc-EEuc)
            cc, pc = [exp(-r * T1) * maximum(c - X1[0], 0.0) - c_coeff1 * (Euc-EEuc),
                      exp(-r * T1) * maximum(X1[0] - c, 0.0) - c_coeff2 * (Euc-EEuc)]
            call_On_Call, put_On_Call = mean(cc), mean(pc)

            # Confidential interval
            loc1, scale1 = norm.fit(cc)
            CI = norm.interval(0.05, loc=loc1, scale=scale1)
            return [call_On_Call, put_On_Call, CI, cc]

        if underlying == 'Put':  # When underlying option is Put
            Eup = maximum(X2 - ST1, 0.) # x pair
            p = bsformula(-1, ST1, X2, r, T2 - T1, sigma2)[0]
            cp, pp = [exp(-r * T1) * maximum(p - X1[1], 0.0), # Y pair
                      exp(-r * T1) * maximum(X1[1] - p, 0.0)]
            VarCov1 = cov(Eup, cp); p_coeff1 = VarCov1[0, 1] / var(Eup)
            VarCov2 = cov(Eup, pp); p_coeff2 = VarCov2[0, 1] / var(Eup)

            # use N to get the real simulated data
            ST1 = S0 * exp(nuT1 + sigmasqrtT1 * randn(1, N))
            Eup = maximum(X2 - ST1, 0.)
            '''calculate the expected value for Euc using Black-Scholes formula'''
            EEup = exp(r * T1) * bsformula(-1, S0, X2, r, T2 - T1, sigma1)[0]
            p = bsformula(-1, ST1, X2, r, T2 - T1, sigma2)[0]
            # control variate --c_coeff1 * (Euc-EEuc)
            cp, pp = [exp(-r * T1) * maximum(p - X1[1], 0.0) - p_coeff1 * (Eup - EEup),
                      exp(-r * T1) * maximum(X1[1] - p, 0.0) - p_coeff2 * (Eup - EEup)]
            call_On_Put, put_On_Put = mean(cp), mean(pp)

            # Confidential interval
            loc2, scale2 = norm.fit(cp)
            CI = norm.interval(0.05, loc=loc2, scale=scale2)
            return [call_On_Put, put_On_Put, CI, cp]

    #def tri_Tree(self):

'''def performance(self):
'''


if __name__=="__main__":
    """Test for Q2"""
    import time
    # Set primary values
    S0, r, T1, T2 = [50.0, 0.025, 1.0, 2.0]; X2 = S0
    # Different client's view
    views = {'Mrs Smith': ([r + 0.03, 0.15], [r + 0.005, 0.3]), 'Mr Johnson': ([r - 0.03, 0.2], [r - 0.01, 0.18]),
             'Ms Williams': ([r - 0.03, 0.18], [r + 0.03, 0.12]), 'Mr Jones': ([r + 0.02, 0.35], [r + 0.02, 0.1]),
             'Miss Brown': ([r + 0.03, 0.15], [r - 0.05, 0.15])}
    names = ('Mrs Smith', 'Mr Johnson', 'Ms Williams', 'Mr Jones', 'Miss Brown')
    ################################################################

    # Question b
    # in different views
    'Question b (1)-Simple Monte Carlo Simulation. ComOpt1 is the prices of compound options'
    ComOpt1 = [0] * len(names); option = ('Call', 'Put')
    'Question b (2)-Smart Lattice method. ComOpt2 is the prices of compound options'
    ComOpt2 = [0] * len(names)
    'Question b (3)-Implicite Finite Difference method. ComOpt3 is the prices of compound options'
    ComOpt3 = [0] * len(names); Smax = 200.; dS = 0.5; dt = 1. / 365
    j = 0
    for v in names:
        mu1, sigma1, mu2, sigma2 = views[v][0][0], views[v][0][1], views[v][1][0], views[v][1][1]
        # At the money
        X1 = [bsformula(1, S0, X2, r, T2 - T1, sigma2)[0],
              bsformula(-1, S0, X2, r, T2 - T1, sigma2)[0]]

        # b (1)-Simple Monte Carlo Simulation
        start_b1 = time.clock()
        print()
        print("Simple Monte Carlo Simulation")
        # For different underlying options
        ComOpt1[j] = [0] * 2; i = 0
        for underlying in option:
            ComOpt1[j][i] = [0] * 4
            ComOpt1[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                            sigma2).simple_Monte_Carlo(underlying, 10000, 777)
            # Get enough size of number to satisfy The Central Limit Theorem
            N = int(floor(std(ComOpt1[j][i][3]) ** 2.0 * 1.96 ** 2.0 * 1002001 /
                          (mean(ComOpt1[j][i][3]) ** 2.0)))
            print("Simulaiton number is", N)
            ComOpt1[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                            sigma2).simple_Monte_Carlo(underlying, N, 777)
            print("For %s, the price of Call on %s and Put on %s are %f and %f"
                  % (v, underlying, underlying, ComOpt1[j][i][0], ComOpt1[j][i][1]))
            print("And the Confidence of Interval is", ComOpt1[j][i][2])
            i += 1
        end_b1 = time.clock()
        print("The running time is : %.03f seconds" % (end_b1 - start_b1))

        # b (2)-Smart Lattice method
        start_b2 = time.clock()
        N = int(365)
        print()
        print("Smart Lattice method")
        ComOpt2[j] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                      sigma2).smart_Lattices_CRR_Bi_Tree(N)
        print("For %s, the prices of Call on Call, Put on Call, Call onp Put, Put on Put are\n"
              "%f, %f, %f, %f"
              % (v, ComOpt2[j][0], ComOpt2[j][1], ComOpt2[j][2], ComOpt2[j][3]))
        end_b2 = time.clock()
        print("The running time is : %.03f seconds" % (end_b2 - start_b2))

        # b (3)-Implicit Finite Difference method
        start_b3 = time.clock()
        print()
        print("Implicit Finite Difference method")
        ComOpt3[j] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                      sigma2).implicit_Finite_Diff(Smax, dS, dt)
        print("For %s, the prices of Call on Call, Put on Call, Call onp Put, Put on Put are\n"
              "%f, %f, %f, %f"
              % (v, ComOpt3[j][0], ComOpt3[j][1], ComOpt3[j][2], ComOpt3[j][3]))
        end_b3 = time.clock()
        print("The running time is : %.03f seconds" % (end_b3 - start_b3))
        j += 1
    #end b question
    ################################################################

    # c -Variance Reduction
    # in different views
    'Question c1-Antithetic Sampling methods. ComOpt4 and ComOpt5 is the prices of compound options'
    ComOpt4 = [0] * len(names); option = ('Call', 'Put')
    j = 0
    for v in names:
        mu1, sigma1, mu2, sigma2 = views[v][0][0], views[v][0][1], views[v][1][0], views[v][1][1]
        # At the money
        X1 = [bsformula(1, S0, X2, r, T2 - T1, sigma2)[0], bsformula(-1, S0, X2, r, T2 - T1, sigma2)[0]]

        print()
        print("Antithetic Sampling")
        # For different underlying options
        ComOpt4[j] = [0] * 2;
        i = 0
        for underlying in option:
            # Antithetic Sampling
            start_c1 = time.clock()
            ComOpt4[j][i] = [0] * 4
            ComOpt4[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                             sigma2).antiSampling(underlying, int(10000), 777)
            # Get enough size of number to satisfy The Central Limit Theorem
            N = int(floor(std(ComOpt4[j][i][3]) ** 2.0 * 1.96 ** 2.0 * 1002001 /
                          (mean(ComOpt4[j][i][3]) ** 2.0)))
            print("Using Antithetic Sampling, simulaiton number is", N)
            ComOpt4[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                             sigma2).antiSampling(underlying, N, 777)
            print("For %s, the price of Call on %s and Put on %s are %f and %f"
                  % (v, underlying, underlying, ComOpt4[j][i][0], ComOpt4[j][i][1]))
            print("And the Confidence of Interval is", ComOpt4[j][i][2])
            i += 1
        end_c1 = time.clock()
        print("The running time is : %.03f seconds" % (end_c1 - start_c1))
        j += 1

    'Question c2-Control Variates methods. ComOpt5 is the prices of compound options'
    ComOpt5 = [0] * len(names); option = ('Call', 'Put')
    j = 0
    for v in names:
        mu1, sigma1, mu2, sigma2 = views[v][0][0], views[v][0][1], views[v][1][0], views[v][1][1]
        # At the money
        X1 = [bsformula(1, S0, X2, r, T2 - T1, sigma2)[0], bsformula(-1, S0, X2, r, T2 - T1, sigma2)[0]]

        print()
        print("Control Variates")
        # For different underlying options
        ComOpt5[j] = [0] * 2;
        i = 0
        for underlying in option:
            # Control Variates
            start_c2 = time.clock()
            ComOpt5[j][i] = [0] * 4
            ComOpt5[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                             sigma2).controlVariates(underlying, int(10000), int(10000), 777)
            # Get enough size of number to satisfy The Central Limit Theorem
            N = int(floor(std(ComOpt5[j][i][3]) ** 2.0 * 1.96 ** 2.0 * 1002001 /
                          (mean(ComOpt5[j][i][3]) ** 2.0)))
            print("Using Control Variates, simulaiton number is", N)
            ComOpt5[j][i] = Comp_Opt_Var_Vol(S0, X1, X2, r, T1, T2, mu1, mu2, sigma1,
                                             sigma2).controlVariates(underlying, N, int(10000), 777)
            print("For %s, the price of Call on %s and Put on %s are %f and %f"
                  % (v, underlying, underlying, ComOpt5[j][i][0], ComOpt5[j][i][1]))
            print("And the Confidence of Interval is", ComOpt5[j][i][2])
            i += 1
        end_c2 = time.clock()
        print("The running time is : %.03f seconds" % (end_c2 - start_c2))
        j += 1
    # end c question
    ################################################################



