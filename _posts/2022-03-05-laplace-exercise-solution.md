---
layout: post
title: Laplace's Method Exercise Solution (Mackay, 2003)
date: 2022-03-04 11:12:00-0400
description: 
categories: machine-learning-posts
---

In this blog, I share my solutions on Mackay 2003 exercises chapter 27 page 342. **Disclaimer: I don't guarantee the validity of each answer. All of the answers are based on my own.** However, I am also open for corrections. If you spot any flaws, feel free to contact me via email: **arijonamarshal@gmail.com** or **marshal.arijona01@ui.ac.id**. All of the notations follow the text book. For further details, please refer to **[https://www.inference.org.uk/itprnn/book.pdf](https://www.inference.org.uk/itprnn/book.pdf)**.

# Problem 1
 A photon counter is pointed
at a remote star for one minute, in order to infer the rate of photons arriving at the counter per minute, $$\lambda$$. Assuming the number of photons collected r has a Poisson distribution with mean $$\lambda$$

\begin{equation}
p(r \vert \lambda) = \exp(\\lambda)\frac{\lambda^r}{r!} \nonumber
\end{equation}


and assuming the improper prior P(λ) = 1/λ, make Laplace approximations to the posterior distribution

(a) over $$\lambda$$ 

(b) over $$\log \lambda$$. [Note the improper prior transforms to $$p(\log \lambda)$$ is
constant.


## Problem 1a

First, let us compute the unnormalized posterior distribution $$p^*(\lambda \vert r)$$ and its log respectively:

$$
\begin{eqnarray}
p^*(\lambda \vert r) &=& p(r \vert \lambda) p(\lambda) \nonumber \\
 &=&  \exp(- \lambda) \frac{\lambda^{r - 1}}{r!} \nonumber
\end{eqnarray}
$$

\begin{equation}
\- \log p^*(\lambda \vert r) = \lambda - (r -1) \log \lambda + \log r! \nonumber
\end{equation}

Recall that we need the mode of distribution in order to perform Laplace approximation. We can compute the mode of $$p^*(\lambda \vert r)$$ via maximum a posteriori (MAP). MAP requires us to find $$\lambda^{\text{MAP}}$$ such that:

\begin{equation}
\lambda^{\text{MAP}} = \underset{\lambda}{\text{min}} - \log p(\lambda \vert r) \nonumber
\end{equation}

we can find MAP by deriving $$- \log p^*(\lambda \vert r)$$, setting it to zero, and finally solving the equation for $$\lambda$$:

$$
\begin{eqnarray}		
\frac{d - \log p^*(\lambda \vert r)}{d\lambda} &=&  1 - \frac{r - 1}{\lambda}= 0 \nonumber \\
\lambda^{\text{MAP}} &=& r - 1 \nonumber 
\end{eqnarray}
$$

Now we can obtain the second order derivative $$c$$ on $$\lambda = \lambda^{\text{MAP}}$$:

$$
\begin{eqnarray}
c &=& \left \vert - \frac{d^2\log p^*(\lambda \vert r)}{d\lambda^2} \right \vert_{\lambda=\lambda^{\text{MAP}}} \nonumber \\ 
&=& -1 \left (\frac{-(r - 1)}{\lambda^2} \right) \nonumber \\
&=& \frac{1}{r - 1} \nonumber
\end{eqnarray} 
$$

We are ready to construct our approximate distribution. First, let's construct the unnnormalized approximation $$q^*(\lambda)$$:

$$
\begin{eqnarray}
q^*(\lambda) &\equiv& p^*(\lambda = \lambda^{\text{MAP}} \vert r) \exp \left(- \frac{c}{2} (\lambda - \lambda^{\text{MAP}})^2 \right) \nonumber \\
&=& p^*(\lambda = \lambda^{\text{MAP}} \vert r) \exp \left(- \frac{1}{2(r - 1)} (\lambda - (r - 1))^2 \right) \nonumber
\end{eqnarray} 
$$

Our normalization factor $$Z_q$$ can be written as:

$$
\begin{eqnarray}
Z_q &=& p^*(\lambda = \lambda^{\text{MAP}} \vert r) \sqrt{\frac{2\pi}{c}} \nonumber \\
&=& p^*(\lambda = \lambda^{\text{MAP}} \vert r) \sqrt{2\pi (r - 1)} \nonumber
\end{eqnarray}
$$

Therefore, we obtain $$q(\lambda) = \frac{1}{\sqrt{2\pi (r - 1)}} \exp \left(- \frac{1}{2(r - 1)} (\lambda - (r - 1))^2 \right) = \mathcal{N}(r - 1, r - 1)$$.

## Problem 1b

From the description, $$\lambda$$ is transformed through the function $$u(\lambda) = \log \lambda$$. Subsequently, the density of $$\lambda$$ is transformed to $$p(u) = p(x) \left \vert \frac{d\lambda}{du}  \right \vert = p(x) \lambda$$. Using this rule, we obtain our unnormalized posterior $$p^*(u(\lambda) \vert r)$$ as follows:

$$
\begin{eqnarray}
p^*(u(\lambda) \vert r) &=& p(u(\lambda)) p(r \vert u(\lambda)) \nonumber \\
&=& \exp(-\lambda)\frac{\lambda^{r}}{r!} \lambda = \exp(-\lambda)\frac{\lambda^{r + 1}}{r!}
\end{eqnarray}
$$

Observe that our unnormalized transformed posterior is just the transforming likelihood since our transformed prior is just a constant. Now, taking the log of $$p^*(u(\lambda) \vert r)$$ gives us:

\begin{equation}
\- \log p^*(u(\lambda) \vert r) = \lambda - (r+1) \log \lambda + \log r!
\end{equation}

Now, let's derive $$u(\lambda)^{\text{MAP}}$$:

$$
\begin{eqnarray}		
\frac{d - \log p^*(u(\lambda) \vert r)}{du(\lambda)} &=&  \exp(\log \lambda) - (r + 1) = 0 \nonumber \\
u(\lambda)^{\text{MAP}} &=& \log (r + 1) \nonumber 
\end{eqnarray}
$$

We then derive our second derivative $$c$$ on $$u(\lambda) = u(\lambda)^{\text{MAP}}$$:

$$
\begin{eqnarray}
c &=& \left \vert - \frac{d^2\log p^*(u(\lambda) \vert r)}{du(\lambda)^2} \right \vert_{u(\lambda)=u(\lambda)^{\text{MAP}}} \nonumber \\ 
&=& \exp(\log (r+1))\nonumber \\
\end{eqnarray}
$$

We are ready to construct our approximate distribution. First, let's construct the unnnormalized approximation $$q^*(u(\lambda))$$:

$$
\begin{eqnarray}
q^*(u(\lambda))  &\equiv& p^*\left(u(\lambda) = u(\lambda)^{\text{MAP}} \vert r \right) \exp \left(- \frac{c}{2} (u(\lambda) - u(\lambda)^{\text{MAP}})^2 \right) \nonumber \\
&=& p^*\left (u(\lambda) = u(\lambda)^{\text{MAP}} \vert r \right) \exp \left(- \frac{(r + 1)}{2} (u(\lambda) - \log (r + 1))^2 \right) \nonumber
\end{eqnarray} 
$$

Our normalization factor $$Z_q$$ can be written as:

$$
\begin{eqnarray}
Z_q &=& p^*\left(u(\lambda) = u(\lambda)^{\text{MAP}} \vert r \right) \sqrt{\frac{2\pi}{c}} \nonumber \\
&=& p^*\left(u(\lambda) = u(\lambda)^{\text{MAP}} \vert r \right) \sqrt{\frac{2\pi} {r + 1}} \nonumber
\end{eqnarray}
$$

Therefore, we obtain $$q(u(\lambda)) = \frac{1}{\sqrt{\frac{2\pi} {(r + 1)}}} \exp \left(- \frac{(r + 1)}{2} (u(\lambda) - \log (r + 1))^2 \right) = \mathcal{N}(\log (r + 1), \frac{1}{r + 1})$$.

# Problem 2
 Use Laplace’s method to approximate the integral

\begin{equation}
Z(u_1, u_2) = \int_{- \infty}^{\infty} f(a)^{u_1} (1 - f(a))^{u_2} da \nonumber
\end{equation}


where $$f(a) = 1/(1 + e−a)$$ and $$u1$$, $$u2$$ are positive. Check the accuracy of the approximation against the exact answer (23.29, p.316) for $$(u1, u2) = (1/2, 1/2)$$ and $$(u1, u2) = (1, 1)$$. Measure the error $$(log Z_p − log Z_q)$$ in bits

**Answer:**

Let us define $$p^*(a) = f(a)^{u_1} (1 - f(a))^{u_2}$$. Therefore, we have $$\log p^*(a) = u_1 \log f(a) + u_2 \log (1 - f(a))$$. Next task is to determine the mode $$a^0$$ of $$p^*(a)$$. Setting the derivative of $$- \log p^*(a)$$ to 0 and solving for $$a$$, we obtain:

$$
\begin{eqnarray}
\frac{d-\log p^*(a)}{da} &=& - \left(\frac{u_1}{f(a)} f'(a) - \frac{u_2}{1 - f(a)} f'(a) = 0 \right) \nonumber \\
&=& - (u_1(1 - f(a)) - u_2 f(a)) \nonumber \\
a^0 &=& - \log \frac{u_2}{u_1} \nonumber
\end{eqnarray}
$$

$$f'(a)$$ refers to $$\frac{df(a)}{da}$$. The second row comes from the fact that $$f'(a) = f(a)(1 - f(a))$$. Next, let us determine the second derivation.

$$
\begin{eqnarray}
\frac{d^2 - \log p^*(a)}{da^2} &=& (u_1 + u_2) f'(a) \nonumber \\
&=& (u_1 + u_2) f(a) (1 - f(a)) \nonumber
\end{eqnarray}
$$

Now, we aim to evaluate $$\frac{d^2 - \log p^*(a)}{da^2}$$ at $$a = a^0$$. Note that we have $$f(a^0) = \frac{u_1}{u_1 + u_2}$$ and $$1 - f(a) = \frac{u_2}{u_1 + u_2}$$. Therefore

$$
\begin{eqnarray}
c = \left \vert \frac{d^2 - \log p^*(a)}{da^2} \right \vert_{a=a^0} = (u_1 + u_2) \frac{u_1}{u_1 + u_2} \frac{u_2}{u_1 + u_2} = \frac{u_1 u_2}{u_1 + u_2} \nonumber \\  
\end{eqnarray}
$$

The normalizing constant can be approximated by:

$$
\begin{eqnarray}
Z_p \simeq Z_q &=& p^*(a^0) \sqrt{\frac{2\pi}{c}} \nonumber \\
&=& \left ( \frac{u_1}{u_1 + u_2} \right )^{u_1} \left(\frac{u_2}{u_1 + u_2} \right )^{u_2}  \sqrt{\frac{2\pi (u_1 + u_2)}{u_1 u_2}} \nonumber \\
\end{eqnarray}
$$

It's time for evaluation !! for $$(u_1, u_2) = (1, 1)$$, we have:

\begin{equation}
Z_q(u_1 = 1, u_2 = 1) = \frac{1}{2} \times \frac{1}{2} \times 2 \sqrt{\pi} = \frac{\sqrt{\pi}}{2} \nonumber 
\end{equation}

\begin{equation}
Z_p(u_1 = 1, u_2 = 1)= \frac{\Gamma(1) \Gamma(1)}{\Gamma(1 + 1)} = 1 \nonumber
\end{equation}

with the error:

\begin{equation}
\log \frac{Z_p}{Z_q} = \log \frac{2}{\sqrt{\pi}} = 0.15 \; \text{bit} \nonumber
\end{equation}

**For the error measurement, we use base 2 logarithm. In other cases we use natural number.**

while for $$(u_1, u_2) = (\frac{1}{2}, \frac{1}{2})$$, we have:

\begin{equation}
Z_q(u_1 = 1/2, u_2 = 1/2) = \frac{1}{2}^{1/2} \times \frac{1}{2}^{1/2} \times \sqrt{8\pi} = \sqrt{2\pi} \nonumber 
\end{equation}

\begin{equation}
Z_p(u_1 = 1/2, u_2 = 1/2)= \frac{\Gamma(1/2) \Gamma(1/2)}{\Gamma(1)} = 1.77^2 = 3.313 \nonumber
\end{equation}

with the error:
\begin{equation}
\log \frac{Z_p}{Z_q} = \log \frac{3.313}{2.5} = 0.12 \; \text{bit} \nonumber
\end{equation}


<!--We also need the second derivative of $$\log p^*(a)$$ on $$a = a^0$$. We can use the chain rule to obtain it. The fact that $$f'(a) = f(a)(1 - f(a))$$ and $$f''(a) = f(a)(1 - f(a))(1 - 2f(a))$$ will make our job easier.

$$
\begin{eqnarray}
\frac{d^2 \log p^*(a)}{da^2} &=& u_1 \left [ \frac{f''(a)}{f(a)} - \frac{f'(a)^2}{f(a)^2} \right] - u_2 \left[ \frac{f''(a)}{1 - f(a)} + \frac{f'(a)^2}{(1 - f(a))^2}\right] \nonumber \\
&=& u_1 \left [ \frac{f(a)f''(a) - f'(a)^2}{f(a)^2} \right] - u_2 \left[ \frac{f''(a) (1 - f(a)) - f'(a)^2}{(1 - f(a))^2} \right] \nonumber \\
&=& -u_1 \left[ (1 - f(a)) f(a)\right] -u_2 \left[ (f(a)(1 - 2f(a))) -  f(a)^2 \right ] \nonumber \\
\end{eqnarray} 
$$

Since we have $$a^0 = - \log \frac{u_2}{u_1}$$, we obtain $$f(a^0) = \frac{u_1}{u_1 + u_2}$$. Now let's substitute $$f(a^0)$$ to the second derivative above:

$$
\begin{eqnarray}
c = \left \vert \frac{d^2 \log p^*(a)}{da^2} \right \vert_{a = a^0} = -u_1 \left[ \frac{u_1 \, u_2}{(u_1 + u_2)^2} \right] -u_2 \left[ \frac{u_1 (u_2 - 2u_1)}{(u_1 + u_2)^2} \right] \nonumber \\
\end{eqnarray}
$$  

Then, the normalizing constant can be approximated by:

$$
\begin{eqnarray}
Z_p \simeq Z_q &=& p^*(a^0) \sqrt{\frac{2\pi}{c}} \nonumber \\
&=& \left ( \frac{u_1}{u_1 + u_2} \right )^{u_1} \left(\frac{u_2}{u_1 + u_2} \right )^{u_2}  \sqrt{\frac{2\pi}{c}} \nonumber \\
\end{eqnarray}
$$

It's time for evaluation !! for $$(u_1, u_2) = (1, 1)$$, we have:

\begin{equation}
Z_q(u_1 = 1, u_2 = 1) = \frac{1}{2} \frac{1}{2} 
\end{equation}

while for $$(u_1, u_2) = (\frac{1}{2}, \frac{1}{2})$$ -->

# Problem 3


Linear regression. $$N$$ datapoints $$\{x^n,t^n\}_{n=1}^{N}$$ are generated by the experimenter choosing each $$x^n$$, then the world delivering a noisy version of the linear function


$$
\begin{eqnarray}
y(x) &=& w0 + w1x \nonumber \\
t^n &\sim& \mathcal{N}(y(x^n), \sigma_\nu^2) \nonumber \\
\end{eqnarray}
$$

Assuming Gaussian priors on $$w_0$$ and $$w_1$$, make the Laplace approximation to the posterior distribution of $$w_0$$ and $$w_1$$ (which is exact, in fact) and obtain the predictive distribution for the next datapoint $$t^{N+1}$$, given $$x^{N+1}$$

**Answer :**

Suppose that $$w = (w_0, w_1)$$, then our prior will be as follows:

$$
\begin{eqnarray}
p(w) &=& \mathcal{N}(w; 0, \mathbf{I}) \nonumber \\
&\propto& \exp\left(- \frac{1}{2} w^{T}w\right) \nonumber \\
\log p(w) &\propto& - \frac{1}{2} w^{T} w \nonumber \\
\end{eqnarray}
$$

<!--Let's just assume both $$p(w_0)$$ and $$p(w_1)$$ are following the standard normal distribution that is $$\mathcal{N}(0, 1)$$ to keep the proble simple.

$$
\begin{eqnarray}
p(w_0) &\propto& \exp\left( \frac{1}{2} w_0^2 \right) \nonumber \\
\log p(w_0) &\propto& \frac{1}{2} w_0^2 \nonumber
\end{eqnarray}
$$

$$
\begin{eqnarray}
p(w_1) &\propto& \exp\left( \frac{1}{2} w_1^2 \right) \nonumber \\
\log p(w_1) &\propto& \frac{1}{2} w_1^2 \nonumber
\end{eqnarray}
$$ -->

Suppose that we have $$(x, t) = \{x^{n}, t^{n}\}_{n=1}^{N}$$ training-data. Since our likelihood is a Gaussian which depends on $$x, w_0,$$ and $$w_1$$, we obtain the likelihood as follows:

$$
\begin{eqnarray}
p(t \vert x, w_0, w_1) &\propto& \prod_{n=1}^{N} \exp \left(-\frac{1}{2\sigma_{\nu}^2} (t^{n} - y(x^{n}))^2 \right) \nonumber \\
&=& \exp\left( -\frac{1}{2\sigma_{\nu}^2} \sum_{n=1}^{N} (t^{n} - y(x^{n}))^2 \right) \nonumber \\
\log p(t \vert x, w_0, w_1) &\propto& - \frac{1}{2\sigma_\nu^2} \sum_{n=1}^{N} (t^{n} - y(x^{n}))^2 \nonumber \\
\end{eqnarray} 
$$

Having prior and likelihood. We are ready to compute the log of unnormalized posterior. We only need to apply the Bayes theorem to do so. 

$$
\begin{eqnarray}
\log p(w \vert x, t) &\propto& \log p(t \vert x, w) + \log p(w) \nonumber \\
&=&  - \frac{1}{2\sigma_\nu^2} \sum_{n=1}^{N} (t^{n} - y(x^{n}))^2 - \frac{1}{2} w^{T} w \nonumber \\
\end{eqnarray}
$$

Let's define the unnormalized posterior as $$p^*(w \vert x, t)$$. The rest of the solution is solving the Laplace approximation. The first step is to obtain $$w^{\text{MAP}}$$ via first derivation of $$- \log p^*(w \vert x, t)$$

$$
\begin{eqnarray}
\frac{d-\log p^*(w \vert x, t)}{dw} &=& 
\begin{bmatrix} 
\frac{d-\log p^*(w \vert x, t)}{dw_0} \\
\frac{d-\log p^*(w \vert x, t)}{dw_1} \\
\end{bmatrix}
\nonumber \\
&=&
\begin{bmatrix}
\frac{-1}{\sigma_\nu^2} \sum_{n=1}^{N} \left[ t^n - (w_0 + w_1 x^n) \right] + w_0\\
\frac{-1}{\sigma_\nu^2} \sum_{n=1}^{N} \left [ t^n - (w_0 + w_1 x^n) \right]x^n + w_1\\
\end{bmatrix}
\nonumber \\
\end{eqnarray}
$$

$$
\begin{eqnarray}
w^{\text{MAP}} &=& 
\begin{bmatrix}
w_0^{\text{MAP}}\\
w_1^{\text{MAP}}\\
\end{bmatrix} 
\nonumber \\
&=& \frac{1}{(n + \sigma_\nu^2)(\sum_{n=1}^N x^n + \sigma_\nu^2) - (\sum_{n=1}^N x^n)^2}
\begin{bmatrix}
(\sum_{n=1}^N (x^n)^2 + \sigma_\nu^2) \sum_{n=1}^N t^n - \sum_{n=1}^N x^n \sum_{n=1}^N x^n t^n \\
\sum_{n=1}^N x^n \sum_{n=1}^N t^n + (n + \sigma_\nu^2) \sum_{n=1}^N x^n t^n\\
\end{bmatrix}
\nonumber \\
\end{eqnarray}
$$

Next step is to obtain the Hessian matrix $$H$$ of $$- \log p^*(w \vert x, t)$$:

$$
\begin{eqnarray}
H &=& 
\begin{bmatrix}
\frac{d^2 - \log p^*(w \vert x, t)}{dw_0^2} & \frac{d^2 - \log p^*(w \vert x, t)}{dw_0 dw_1} \\
\frac{d^2 - \log p^*(w \vert x, t)}{dw_1 dw_0} & \frac{d^2 - \log p^*(w \vert x, t)}{dw_1^2}
\end{bmatrix}
\nonumber \\
&=&
\begin{bmatrix}
\frac{n}{\sigma_\nu^2} + 1 & \frac{\sum_{n=1}^{N} x^n}{\sigma_\nu^2}\\
\frac{\sum_{n=1}^{N}x^n}{\sigma_\nu^2} & \frac{\sum_{n=1}^{N} (x^n)^2}{\sigma_\nu^2} + 1 \\
\end{bmatrix}
\nonumber \\
\end{eqnarray}
$$

Now, we can approximate $$p^*(w \vert x, t)$$ with a distribution $$q^*(w)$$:
$$
\begin{eqnarray}
q^*(w) = p^*(w^{\text{MAP}}) \exp \left[\frac{-1}{2} (w - w^{\text{MAP}})^T H (w - w^{\text{MAP}}) \right] \nonumber \\
\end{eqnarray}
$$

Subsequently, we obtain the normalization factor $$Z_q$$

$$
\begin{eqnarray}
Z_q = p^*(w^{\text{MAP}}) \sqrt{\frac{(2\pi)^K}{\text{det} H}} \nonumber \\
\end{eqnarray}
$$

with $$K$$ denotes the dimensionality of $$w$$. Finally our approximate distribution follows a normal distribution

$$
\begin{eqnarray}
q(w) &=& \frac{q^*(w)}{Z_q} \nonumber \\
&=& \frac{1}{\sqrt{\frac{(2\pi)^2}{\text{det H}}}} \exp \left[\frac{-1}{2} (w - w^{\text{MAP}})^T H (w - w^{\text{MAP}}) \right] \nonumber \\
 &=& \mathcal{N}(w; w^{\text{MAP}}, H) \nonumber \\
\end{eqnarray}
$$

Given a new data point $$(x^{N+1})$$, we compute the approximate predictive distribution:

$$
\begin{eqnarray}
p(t^{N+1} \vert x^{N+1}) =  \int p(t^{N+1} \vert x^{N + 1}, w) q(w) dw \nonumber \\
\end{eqnarray}
$$

However, compute this integral is often intractable. We can utilize Monte Carlo estimation to simplify the computation.

$$
\begin{eqnarray}
p(t^{N+1} \vert x^{N+1}) \approx \sum_{i=1}^{M} p(t^{N+1} \vert x^{N + 1}, w^{i}), \qquad w^{i} \sim q(w) \nonumber \\  
\end{eqnarray}
$$

with M is the number of samples $$w$$. The more samples we use the more accurate is the prediction.