---
layout: post
title: Probability Improvement and Expected Improvement Acquisition Function
date: 2023-10-05 00:00:00-0400
description:
tags: acquisition-function Bayesian-optimization
categories: machine-learning-posts
---

This post assumes the reader has a prequisite knowledge about Bayesian optimization. Recall that Bayesian optimization is a zero-th order optimization method which aims to find the optimum $$x^\ast$$ of a black-box function $$f: \mathcal{X} \rightarrow \mathbb{R}$$. Bayesian optimization requires two important components, that is the surrogate model and the acquisition function. Due to black-box nature of $$f$$, we introduce a surrogate function and instead perform the optimization w.r.t. this new function. In common settings, we utilize a Gaussian process (GP) as our surrogate model. The motivation comes from the Bayesian philosophy where we start with a belief and iteratively update it as we encounter data which explains the true distribution called the posterior distribution. 

This post is focused on the second component, that is the acquisition function. This function governs the next input $$x_t$$ to be evaluated, with $$t$$ denotes the particular time step of the acquisition. The acquired data has the optimum statistical properties w.r.t. our surrogate model, i.e. the expectation, entropy, etc. Now, let us narrowing our scope again into two specific acquisition functions, that is the probability improvement and the expected improvement. 

### Probability Improvement

We first give the definition of improvement. Given the incummbent best value $$x^\ast \in \mathcal{X}$$ and an arbitrary input $$x \in \mathcal{X}$$, we define the improvement $$I(x)$$ as

$$
\begin{eqnarray}
I(x) = \max(f(x) - f(x^\ast), 0)
\end{eqnarray}
$$

Note that we abuse the notation $$x^\ast$$ for a while as previously $$x^\ast$$ denotes the global optimum of the function $$f$$. It is obvious that $$I(x) \in [0, +\infty] \quad \forall x \in \mathcal{X}$$ and $$x = x^\ast \rightarrow I(x) = 0$$. Recall that we employ GP as the proxy of $$f$$. For a particular data $$x$$, the function value $$f(x)$$ follows a Gaussian distribution

$$
f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))
$$

Commonly we perform reparameterization-trick to draw samples from $$f(x)$$. First, we introduce a random variable $$z$$ drawn from a standard normal distribution $$\mathcal{N}(0, 1)$$. It is known that drawing samples from such distribution is relatively easy. Leveraging this random variable, we obtain a new sample $$f(x) = \mu(x) + \sigma(x) z$$. Substituting the new definition will give us

$$
I(x) = \max(f(x) - f(x^\ast)) = \max(\mu(x) + \sigma(x)z - f(x^\ast), 0) \quad z \sim \mathcal{N}(0, 1)
$$

The probability improvement evaluates how likely the candidate $$x$$ gives us a positive improvement. Recall that we evaluate the probability w.r.t. $$\mathcal{N}(\mu(x), \sigma^2(x))$$. Mathematically, we can write $$\mathrm{PI}(x)$$ as

$$
\mathrm{PI}(x) = p(I(x) > 0) \iff p(f(x) > f(x^\ast))
$$

By applying the additive and constant scaling properties of normal distribution, we obtain the following analytical form

$$
\mathrm{PI}(x) = 1 - \Phi(z_0) = \Phi(- z_0) = \Phi\left( \frac{\mu(x) - f(x^\ast)}{\sigma(x)} \right)
$$

with $$\Phi(z) \triangleq \mathrm{CDF}(z)$$ and $$z_0 = \frac{f(x^\ast) - \mu(x)}{\sigma(x)}$$

### Expected Improvement
Unlike the probability improvement, the expected improvement (EI) (as the name suggests) aims to evaluate the expected value of $$I(x)$$ over $$f$$. Intuitively, this criterion evaluates the average magnitude of the improvement.

$$
\begin{eqnarray}
\mathrm{EI}(x) \triangleq \mathbb{E}[I(x)] = \int_{- \infty}^\infty I(x) \phi(z)
\end{eqnarray}
$$

Substituting the definition of probability improvement, we then obtain

$$
\mathrm{EI} = \int_{- \infty}^\infty I(x) \phi(z) = \int_{- \infty}^\infty \max(f(x) - f(x^\ast), 0) \phi(z) dz
$$

In order to compute the integral, we need to get rid of the $$\max$$ operator. First, we decompose the integral into two parts. The first part is where $$I(x) \leq 0$$ and the later part is where $$I(x) > 0$$. To set the bound for each integral, recall that we can perform the reparameterization trick to rewerite $$f(x)$$, that is $$f(x) = f(x^\ast) \rightarrow \mu + \sigma z = f(x^\ast) \rightarrow z_0 = \frac{f(x^\ast) - \mu}{\sigma}$$. Thus, we can write $$\mathrm{EI}(x)$$ as

$$
\mathrm{EI}(x) = \int_{- \infty}^{z_0} I(x) \phi(z) \, dz + \int_{z_0}^\infty I(x) \phi(z) \, dz
$$

Observe that the first term vanishes to $$0$$ since $$\forall z \leq z_0$$ we have $$I(x) = 0$$. Therefore, we only need to evaluate the second part of the integral.

$$ 
\begin{aligned}
\mathrm{EI}(x) &= \int_{z_0}^\infty \max(f(x) - f(x^\ast), 0) \phi(z) \, dz = \int_{z_0}^\infty \mu(x) + \sigma(x) z - f(x^\ast) \phi(z) \, dz \\
&= \int_{z_0}^\infty (\mu - f(x^\ast)) \phi(z) dz + \int_{z_0}^\infty \sigma z \frac{1}{\sqrt{2 \pi}} \exp \left( \frac{-1}{2} z^2 \right ) dz \\
&= (\mu - f(x^\ast)) \int_{z_0}^\infty \phi(z) dz + \frac{\sigma}{\sqrt{2 \pi}} \int_{z_0}^\infty z \exp \left( \frac{-1}{2} z^2 \right) dz \\
&= (\mu - f(x^\ast)) (1 - \Phi(z_0)) - \int_{z_0}^\infty \left( \exp \left( \frac{-1}{2} z^2 \right) \right)^\prime dz \\
&= (\mu - f(x^\ast)) (1 - \Phi(z_0)) - \frac{\sigma}{\sqrt{2 \pi}} \left[ \exp\left(\frac{-1}{2} z^2\right) \right]_{z_0}^\infty \\
&= (\mu - f(x^\ast)) (1 - \Phi(z_0)) + \sigma \phi(z_0) \\
&= (\mu - f(x^\ast)) \Phi\left( \frac{\mu - f(x^\ast)}{\sigma} \right) + \sigma \phi\left( \frac{\mu - f(x^\ast)}{\sigma} \right) 
\end{aligned} 
$$

The last row comes from the fact that the normal density is symmetric, i.e., $$\phi(z_0) = \phi(- z_0)$$. $$\mathrm{EI}(x)$$ takes high value when $$\mu > f(x^\ast)$$ As a side note, $$\mathrm{EI}$$ requires the uncertainty $$\sigma > 0$$ since $$\sigma = 0 \rightarrow \mathrm{EI}(x) = 0$$. Finally, we intoroduce a hyperparameter $$\xi$$ which control the degree of exploration

$$
\begin{eqnarray}
\mathrm{EI}(x; \xi) = (\mu - f(x^\ast) - \xi) \Phi\left( \frac{\mu - f(x^\ast) - \xi}{\sigma} \right) + \sigma \phi\left( \frac{\mu - f(x^\ast) - \xi}{\sigma} \right) 
\end{eqnarray}
$$

Note that $$\xi = 0 \rightarrow \mathrm{EI}(x; \xi) = \mathrm{EI}(x)$$


###### **References**

- Kamperis, S. (2021) Acquisition functions in Bayesian Optimization, https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html.
