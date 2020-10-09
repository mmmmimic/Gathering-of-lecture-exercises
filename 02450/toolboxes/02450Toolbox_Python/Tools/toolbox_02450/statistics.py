import scipy.stats
import numpy as np
import scipy.stats as st

def correlated_ttest(r, rho, alpha=0.05):
    rhat = np.mean(r)
    shat = np.std(r)
    J = len(r)
    sigmatilde = shat * np.sqrt(1 / J + rho / (1 - rho))

    CI = st.t.interval(1 - alpha, df=J - 1, loc=rhat, scale=sigmatilde)  # Confidence interval
    p = 2*st.t.cdf(-np.abs(rhat) / sigmatilde, df=J - 1)  # p-value
    return p, CI


def jeffrey_interval(y, yhat, alpha=0.05):
    m = sum(y - yhat == 0)
    n = y.size
    a = m+.5
    b = n-m + .5
    CI = scipy.stats.beta.interval(1-alpha, a=a, b=b)
    thetahat = a/(a+b)
    return thetahat, CI

def ttest_onemodel(y_true, yhat, loss_norm_p=1, alpha=0.05):
    # perform statistical comparison of the models
    # compute z with squared error.

    zA = np.abs(y_true - yhat) ** loss_norm_p
    CI = st.t.interval(1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA))
    return np.mean(zA), CI

def ttest_twomodels(y_true, yhatA, yhatB, alpha=0.05, loss_norm_p=1):
    zA = np.abs(y_true - yhatA) ** loss_norm_p
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zB = np.abs(y_true - yhatB) ** loss_norm_p

    z = zA - zB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return np.mean(z), CI, p


def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p


if __name__ == "__main__":
    z = [-9.94147773e+02, 1.26057137e+02, 2.43068571e+03, 1.34943873e+02, -7.32103331e+02, 2.37564709e+02,
           2.50241916e+02, 2.57480953e+02, -2.63697057e+02, -6.87957076e+01, -2.79913347e+02, 1.88978039e+02,
           1.98121892e+02, 5.41920321e+01, 1.70814489e+02, 9.50546024e+02, -1.42327811e+02, 1.76465996e+02,
           6.87389306e+01, 1.73725613e+03, -1.78676140e+02, 1.52421405e+03, -5.30574002e+01, 1.95582309e+00,
           -1.94314010e+02, -6.72125537e+02, 1.62167916e+02, 1.78461753e+02, -1.24817459e+02, 1.43904422e+02,
           2.45598432e+02, 4.17515769e+02, 1.34710476e+02, -4.48734895e+01, 1.05674612e+02, -3.39105804e+02,
           -5.34365506e+02, 2.23486078e+02, 1.97750315e+02, -3.00557776e+03, 9.63587836e+01, -1.85012667e+02,
           2.54862222e+02, -1.78881284e+02, -1.03805766e+02, 2.52354768e+02, -6.00848307e+02, 3.71357436e+00,
           2.38950633e+02, -1.88401811e+03, 1.86325333e+02, 2.45168149e+02, 1.14115851e+01, 1.18459847e+02,
           4.20244456e+02, -1.96854780e+02, -1.24603029e+01, -5.54211898e+02, -1.57707245e+01, -5.39761905e+02,
           -2.82533665e+02, 1.42265335e+02, 1.30362591e+02, 3.63309122e+01, 1.38202398e+02, 1.58929137e+02,
           1.58929177e+02, 7.10797177e+02, 1.34089160e+01, 9.32132688e+02, 3.46853860e+01, 6.27785220e+01,
           2.81806999e-02, -1.52944174e+02, 2.66329889e+02, 1.62190118e+02, -3.89048944e-03, -2.60694426e+02,
           -7.15940302e+02, 2.25831089e+02, -1.77851578e+01, 2.66329889e+02, 1.08980992e+03, 1.56404585e+02,
           2.66329889e+02, 6.63044600e+02, 8.08266552e+01, 1.83926579e+02, 1.77769644e+02, -5.92678110e+01,
           1.86044032e+02, 1.59819830e+02, 2.60035987e+02, 1.60910872e+02, -2.39925571e+02, -1.03542616e+02,
           -1.30351275e+01, 3.88166963e+03, 1.51075198e+02, -1.65484521e+02, 9.08165687e+01, 1.18686751e+03,
           1.65290154e+02, -1.91692974e+02, 2.75584781e+02, -1.91227724e+03, -9.14883857e+00, -6.03404163e+01,
           1.26539212e+02, 5.32728542e+01, 7.13462504e+02, 2.24593771e+02, 1.16993301e+02, 1.08405310e+02,
           5.76378276e+01, 1.27516156e+02, 1.93353908e+01, 2.75555832e+02, -8.77754648e+01, -3.75658826e+02,
           -7.52816578e+02, -4.34021742e+02, 5.95930150e+01, 9.43829397e+02, -4.37258761e+02, 1.27857209e+02,
           4.36410358e+01, -9.96612122e+01, 2.24738210e+03, 1.60453092e+02, 2.03273360e+02, -8.06696669e+01,
           9.88763264e+01, 5.55727999e+02, -2.18588047e+02, 1.91855517e+02, 1.26188907e+03, -6.70477718e+02,
           -3.28242036e+02, 4.25807472e+01, 2.87933046e+03, 1.28770056e+03, 1.77890518e+02, 9.42159762e+02,
           1.97441517e+02, 6.71145887e+01, 1.97441517e+02, 1.38789855e+02, 2.30957514e+02, -1.18130059e+02,
           -1.09434948e+02, -3.46961432e+02, 1.25455407e+02, -1.97299428e+03, 1.77283165e+02, -3.36631354e+02,
           -2.60743339e+01, -2.24421069e+02, 1.95480316e+02, 3.54171629e+02, 1.65461586e+02, 1.05668384e+02,
           1.67418017e+01, -8.44526008e+02, 2.58552624e+02, 2.56605849e+02, 1.91315916e+02]

    alpha = 0.05
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

    print(p)
    print(CI)
    a = 123
