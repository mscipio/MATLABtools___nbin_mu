# MATLAB tools: nbin*_mu

## Negative Binomial Distribution
Negative binomial regression is for modeling count variables, usually for over-dispersed count outcome variables.

##### Negative binomial regression
Negative binomial regression can be used for over-dispersed count data, that is when the conditional variance exceeds the conditional mean. It can be considered as a generalization of Poisson regression since it has the same mean structure as Poisson regression and it has an extra parameter to model the over-dispersion. If the conditional distribution of the outcome variable is over-dispersed, the confidence intervals for the Negative binomial regression are likely to be narrower as compared to those from a Poisson regression model.

##### Poisson regression
Poisson regression is often used for modeling count data. Poisson regression has a number of extensions useful for count models.

##### Zero-inflated regression model
Zero-inflated models attempt to account for excess zeros. In other words, two kinds of zeros are thought to exist in the data, “true zeros” and “excess zeros”. Zero-inflated models estimate two equations simultaneously, one for the count model and one for the excess zeros.

##### OLS regression
Count outcome variables are sometimes log-transformed and analyzed using OLS regression. Many issues arise with this approach, including loss of data due to undefined values generated by taking the log of zero (which is undefined), as well as the lack of capacity to model the dispersion.

Matlab provides some functions to experiments with Negative Binomial Distribution.

Problem is that, for this parcticular family of distribution, you can find different kind of parametrization. According to the problem you are trying to solve or reproduce, one parametrization can me better than another.

For a general idea of what I mean by "different parametrization", you can have a look at the Wikipedia page related to NB distribution, at https://en.m.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations.

## Matlab choice of parametrization: r-p

In its simplest form (when r is an integer), the negative binomial distribution models the number of failures x before a specified number of successes is reached in a series of independent, identical trials. Its parameters are the probability of success in a single trial, p, and the number of successes, r. A special case of the negative binomial distribution, when r = 1, is the geometric distribution, which models the number of failures before the first success.

More generally, r can take on non-integer values. This form of the negative binomial distribution has no interpretation in terms of repeated trials, but, like the Poisson distribution, it is useful in modeling count data. The negative binomial distribution is more general than the Poisson distribution because it has a variance that is greater than its mean, making it suitable for count data that do not meet the assumptions of the Poisson distribution. In the limit, as r increases to infinity, the negative binomial distribution approaches the Poisson distribution.

To deal with this version of negative binomial distribution, the **Statistics and Machine Learning Toolbox** provide the following set of functions:

- *nbinrnd.m*
- *nbinlike.m*
- *nbinfit.m*
- *nbinpdf.m*


## Ecological parameterization of the negative binomial: µ-k
The “ecological” parameterization of the negative binomial replaces the parameters **p** (probability of success per trial) and **n** (number of successes before you stop counting failures) with ***µ = n(1−p)/p***, the mean number of failures expected (or of counts in a sample), and **k**, which is typically called an **overdispersion** parameter. 

Confusingly, **k** is sometimes called *size*, because it is mathematically equivalent to **n** in the failure-process parameterization.

The overdispersion parameter measures the amount of clustering, or aggregation, or heterogeneity, in the data: a smaller **k** means more heterogeneity. The variance of the negative binomial distribution is ***µ+µ^2/k***, and so as **k** becomes large the variance approaches the mean and the distribution approaches the Poisson distribution. For *k > 10*, the negative binomial is hard to tell from a Poisson distribution, but **k** is often less than 1.
.
Specifically, you can get a negative binomial distribution as the result of a Poisson sampling process where the rate **λ **itself varies. If the distribution of **λ** is a gamma distribution with shape parameter **k** and mean **µ**, and **x** is Poisson-distributed with mean **λ**, then the distribution of **x** be a negative binomial distribution with mean **µ** and overdispersion parameter **k** (May, 1978; Hilborn and Mangel, 1997). In this case, the negative binomial
reflects unmeasured (“random”) variability in the population.

While available in R, this kind of parametrization is not provided by Matlab most standard libraries, so this repo is about adding them so that you can have a choice of the best version of NB distribution you want to use.

In detail, the new versions of the Matlab files you can find here are the following:

- *nbinrnd_mu.m*
- *nbinlike_mu.m*
- *nbinfit_mu.m*
- *nbinpdf_mu.m*

