---
layout: post
title: A Review of Periodic Activation Function Induce Stationarity (Under Construction)
date: 2022-07-19 11:12:00-0400
description:
tags: periodic-activation-function Bayesian-neural-network uncertainty-quantification
categories: machine-learning-posts
---

During this summer, I work as a research assistant in Arno Solin's lab (Dep. of CS Aalto University). My duty is to investigate the paper titled "Periodic Activation Function induce stationarity". The main idea of this paper is to show that a BNN equipped with a periodic activation function will behave like a stationary kernel of Gaussian process, e.g. Matern Kernel, RBF kernel, etc. The stationarity term here means that our NN is translation-invariant, pushing the BNN to put the uncertainty only based on the distance of data. Please check the [paper](https://arxiv.org/pdf/2110.13572.pdf) for further details.


#### Sin Activation Function
$$\begin{eqnarray}
\sigma(x) = \sqrt{2} \sin(x) \\
p(b) = \text{Uniform}[-\pi, \pi] 
\end{eqnarray}$$

The covariance function then can be written as follows:

\begin{equation} \label{eq:sinactivationfunction}
k(x, x^\prime) = \int p(w) \frac {1}{\pi} \int_{-\pi}^\pi \sin(wx + b) \sin(wx^\prime + b) \, db \, dw
\end{equation}

Note that 2 is canceled out due to coefficient of $$\sigma$$. Now, let us solve the inner integral:

$$
\begin{eqnarray}
&& \int_{-\pi}^\pi \sin(wx + b) \sin(wx^\prime + b) \, db \\
&=&	 \int_{-\pi}^\pi \frac{1}{2} (\cos(wx + b - wx^\prime - b) - \cos(wx + b + wx^\prime + b)) \, db \\
&=& \frac{1}{2} \int_{-\pi}^\pi \cos(w(x -x^\prime)) - \cos(w(x + x^\prime) + 2b) \, db \\
&=& \frac{1}{2} [\cos(w(x -x^\prime)) b]_{- \pi}^{\pi} = \pi \cos(w(x -x^\prime))
\end{eqnarray}
$$

We obtained the second row by applying the property $$\sin(x) \sin(y) = 1/2 (\cos(x - y) - \cos(x + y))$$. At the last row, we ignore the second term due to its periodicity variable $$2b$$ (symmetric). Now, we plug the last row back to Equation \eqref{eq:sinactivationfunction} to get the following:

\begin{equation}
k(x, x^\prime) = \int p(w) \cos(w(x - x^\prime)) dw
\end{equation}

Observe that we can write $$\cos(x)$$ in terms of natural exponentiation via Euler's formula, that is $$\cos(x) = 1/2 (\exp(ix) + \exp(-ix))$$, where $$i$$ denotes imaginary number $$\sqrt{- 1}$$. Thus, we obtain

$$
\begin{equation}
k(x, x^\prime) = \frac{1}{2} \int p(w) (\exp(iw(x - x^\prime)) + \exp(-iw(x - x^\prime))) \, dw
\end{equation}
$$

Assume that the prior is symmetric on $$w$$, we can perform change of variables $$w = -w$$ to obtain

$$
\begin{eqnarray}
k(x, x^\prime) &=&\frac{1}{2} \int p(w) \exp(iw(x - x^\prime)) \, dw + \frac{1}{2} \int p(-w) \exp(iw(x - x^\prime)) \, dw \\
&& = \int p(w) \exp(iwr) \, dw 
\end{eqnarray}
$$ 

Where $$r = x - x^\prime$$. Observe that $$p(w) = \frac{1}{2\pi} S(w)$$, which is want to be shown.

#### Sin Cos Activation Function

\begin{equation}
\sigma(x) = \sin(x) + \cos(x)
\end{equation}

Let us assume that our single neural network is bias free. Given the activation function above, we can write the covariance function $$k(x, x^\prime)$$ as follows:

$$
\begin{eqnarray}
k(x, x^\prime) &=& \int p(w) (\sin(wx) + \cos(wx)) (\sin(wx^\prime) + \cos(wx^\prime)) \, dw \\
&=& \int p(w) (\sin(wx) \sin(wx^\prime) + \sin(wx) \cos(wx^\prime) \nonumber \\ 
&\quad& + \cos(wx) \sin(wx^\prime) + \cos(wx) \cos(wx^\prime)) \, dw \\
&=& \int p(w) (\cos(wx - wx^\prime) + \sin(wx + wx^\prime)) \, dw \\
\end{eqnarray}
$$

We apply distributed rule on the first row to obtain the second row. Subsequently, applying indentitites $$\sin(x \pm y) = \sin(x) \cos(y) \pm \cos(x) \sin(y)$$ and $$\cos(x \pm y) = \cos(x) \cos(y) \mp \sin(x) \sin(y)$$ on the second row gives us the third row. Rewriting $$\sin$$ and $$\cos$$ in Euler's form gives us

$$
\begin{eqnarray}
k(x, x^\prime) &= \frac{1}{2} \int p(w) \exp(iw(x - x^\prime)) \, dw + \frac{1}{2} \int p(w) \exp(-iw(x - x^\prime)) \, dw \\
&+ \frac{1}{2} \int p(w) i\exp(-iw(x + x^\prime)) \, dw - \frac{1}{2} \int p(w) i\exp(iw(x + x^\prime)) \, dw
\end{eqnarray}
$$

Since the support of $$p(w)$$ is on the entire real line and with the assumption that $$p(w)$$ is symmetric, we obtain

$$
\begin{eqnarray}
k(x, x^\prime) &= \frac{1}{2} \int p(w) \exp(iw(x - x^\prime)) \, dw  + \frac{1}{2} \int p(w) \exp(-iw(x - x^\prime)) \, dw = \int p(w) \exp(iwr) \, dw \\
\end{eqnarray}
$$

Again with $$r = x - x^\prime$$

#### Triangle Wave Activation

We can write triangle wave activation as a parametric function

\begin{equation}
	\psi(x) = \frac{4}{p} \left( x - \frac{p}{2} \lfloor	\frac{2x}{p} + \frac{1}{2} \rfloor \right) (-1)^{\lfloor\frac{2x}{p} + \frac{1}{2}\rfloor}
\end{equation}

$$p$$ is responsible to control the period of the function. The authors of the paper chose $$p=2\pi$$. Now to make the analysis becomes feasible, we need to perform Fourier series approximation on $$\psi(x)$$.

\begin{equation}
\psi(x) = \underset{n \rightarrow \infty}{\lim} \frac{8}{\pi^2} \sum_{k=0}^{n - 1} (-1)^k (2k + 1)^{-2} \sin((2k + 1)x)
\end{equation}

Let us assume that $$\lambda_k := 2k + 1$$ and the bias is sampled from an uniform distribution. Thus, we can write our activation function as follows

$$
\begin{eqnarray}
&&\sigma(z) = \sqrt{2} \sum_{k=0}^{n - 1} (-1)^k \lambda_k^{-2} \sin(\lambda_k z) \\
&&p(b) = \text{Uniform}(-\pi, \pi)
\end{eqnarray}
 $$
 
Note that $$\sigma$$ will converge to $$\psi$$ as we $$n$$ goes to infinity. Subsequently, the corresponding covariance function $$k(x, x^\prime)$$ will be as follows
 
 $$
 \begin{eqnarray}
 k(x, x^\prime) &&= \int p(w) \int 2p(b) [\sum_{k=0}^{n - 1} (-1)^k \lambda_k^{-2} \sin(\lambda_k (wx + b))]   \nonumber \\
 && \quad [\sum_{j=0}^{n - 1} (-1)^j \lambda_j^{-2} \sin(\lambda_j (wx^\prime + b))] \, dw \, db
 \end{eqnarray}
 $$
 
 For now, let us solve the inner integral for the case $$k \neq j$$:
 
  $$
 \begin{eqnarray}
 &&\int \frac{1}{\pi} \frac{(-1)^{k + j}}{\lambda_k^2 \lambda_j^2} [\sin(\lambda_k (wx + b))]  [\sin(\lambda_j (wx^\prime + b))] \, db  \nonumber \\
 &&= \frac{1}{2\pi} \frac{(-1)^{k + j}}{\lambda_k^2 \lambda_j^2} \int \cos(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j)) - \cos(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j)) \, db \nonumber \\
 &&= \frac{1}{2\pi} \frac{(-1)^{k + j}}{\lambda_k^2 \lambda_j^2} [\frac{\sin(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j))}{\lambda_k - \lambda_j} - \frac{\sin(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j))}{\lambda_k + \lambda_j}]_{-\pi}^{\pi} \\
 &&= 0 \nonumber
 \end{eqnarray}
 $$
 
 Note that the above integral resulted in $$0$$ since both $$\lambda_k, \lambda_j$$ are odd and the magnitude of the lower bound and the upper bound of integral is equal. Therefore, the remaining terms are the case where $$k = j$$. We then rewrite the covariance function as follows:
 
  $$
 \begin{eqnarray}
 k(x, x^\prime) &&= \int p(w) \int 2p(b) \sum_{k=0}^{n - 1} \frac{(-1)^{2k}}{\lambda_k^4} \sin(\lambda_k (wx + b)) \sin(\lambda_k (wx^\prime + b)) \, dw \, db
 \end{eqnarray}
 $$
 
 Again, let us solve the inner integral for each summand. For now, we take the constant out of the inner integral and will involve them in the outer integral. 
 
 $$
 \begin{eqnarray}
 &&\int_{-\pi}^{\pi} 2 p(b) \sin(\lambda_k (wx + b)) \sin(\lambda_k (wx^\prime + b)) db \nonumber \\
 &&= \int_{-\pi}^{\pi} \frac{1}{2 \pi} [\cos(w\lambda_k (x -  x^\prime)) - \cos(w\lambda_k (x + x^\prime) + 2b\lambda_k)] \, db \nonumber \\
 &&=\cos(w\lambda_k (x -  x^\prime)) 
 \end{eqnarray}
 $$
 
 Note that the second term in the second row is canceled out due to its even shifting coefficient and equal magnitude of lower and upper bound. Let us plug the result back to the covariance function.
  
\begin{equation}
 	k(x, x^\prime) = \int p(w) \sum_{k=0}^{n - 1} \frac{(-1)^{2k}}{\lambda_k^4} \cos(w\lambda_k (x -  x^\prime)) dw
\end{equation}
 
 In order to get the exact solution, we take the limit $$n \rightarrow \infty$$. 
 
 \begin{equation}
 	k(x, x^\prime) = \int p(w) \underset{n \rightarrow \infty}{\lim} \sum_{k=0}^{n - 1} \frac{1}{\lambda_k^4} \cos(w\lambda_k (x -  x^\prime)) dw
\end{equation}

The equation above is based on the fact that $$(-1)^{2k} = 1$$ for $$k \in \mathbb{N} \geq 0$$. Next, we need the dominated convergence theorem to take the limit outside of the integral. Let $$f(n) = p(w) \sum_{k=0}^{n - 1} \frac{1}{\lambda_k^4} \cos(w\lambda_k (x -  x^\prime))$$. Then, it requires another function $$g(n)$$ s.t. $$\vert f_n(w) \vert \geq g(w), \forall n$$ and $$\int g(w) dw < \infty$$

$$
\begin{eqnarray}
\vert f_n(w) \vert &&= \sum_{k=0}^{n - 1} \frac{1}{(2k + 1)^4} \cos(w \lambda_k (x -  x^\prime)) \\
&& \leq p(w) \sum_{k=0}^{n - 1} \frac{1}{(2k + 1)^4} \\
&& \leq p(w) \sum_{k=0}^{n - 1} \frac{1}{k^4} = \frac{\pi ^ 4}{90} p(w)
\end{eqnarray}
$$

Furthermore, we have $$\int \pi^4/90 p(w) dw = \pi^4 / 90 < \infty$$. Since $$g(w)$$ satisfies all constraints, we can rewrite $$k(x, x^\prime)$$ as follows:

\begin{equation}
k(x, x^\prime) = \underset{n \rightarrow \infty}{\lim} \int p(w) \sum_{k=0}^{n - 1} \frac{1}{\lambda_k^4} \cos(w \lambda_k (x -  x^\prime)) dw
\end{equation}

We also can rewrite the above equation in the form of mixture density(assuming the density $$p$$ is a member of location-scale family, which is the case in this paper). By applying the Euler's form we can write the covariance function as follows:

$$
\begin{eqnarray}
k(x, x^\prime) &&= \underset{n \rightarrow \infty}{\lim} \int \sum_{k=0}^{n - 1} p(w) \frac{1}{\lambda_k^4} \exp(i \lambda_k w (x -  x^\prime)) \, dw \\
&& = \underset{n \rightarrow \infty}{\lim} \int \sum_{k=0}^{n - 1} p(w) \pi_k \exp(i \lambda_k w (x -  x^\prime)) \, dw \\
&& = \underset{n \rightarrow \infty}{\lim} \int \sum_{k=0}^{n - 1} p(w \vert \lambda_k) \pi_k \exp(iw (x -  x^\prime)) \, dw \\
\end{eqnarray}
$$

Let $$\hat{p}(w) = \underset{n \rightarrow \infty}{\lim} \sum_{k=0}^{n - 1} p(w \vert \lambda_k) \pi_k$$ and $$r = x - x^\prime$$, we recover the Wiener-Kinchin theorem

\begin{equation}
k(x, x^\prime) = \int \hat{p}(w) \exp(iwr) \, dw
\end{equation}

#### Periodic ReLU Activation

We can write the periodic ReLU function as a sum of triangle wave activation function with the second term is shifted by half a period from the first term. 

\begin{equation}
\psi(x) = \frac{2}{\pi} (((x + \frac{\pi}{2}) - \pi \lfloor \frac{(x + \frac{\pi}{2})}{\pi} + \frac{1}{2} \rfloor) (-1)^{\lfloor \frac{(x + \frac{\pi}{2})}{\pi} + \frac{1}{2} \rfloor} + (x - \pi \lfloor \frac{x}{\pi}  + \frac{1}{2} \rfloor) (-1)^{\lfloor \frac{x}{\pi}  + \frac{1}{2} \rfloor})
\end{equation}

Again, we approximate $$\psi(x)$$ through Fourier transformation to obtain

\begin{equation}
\sigma(x) = \underset{n \rightarrow \infty}{lim} \sum_{k=0}^{n - 1} (-1)^k \lambda_k^{-2} (\sin(\lambda_k (x + \frac{\pi}{2}))  + \sin(\lambda_k x))
\end{equation}

With $$p(b) = \text{Uniform}[-\pi, \pi]$$. Given the above activation function, we can define the covariance function as follows

$$
\begin{eqnarray}
k(x, x^\prime) &&= \int p(w) \int p(b) [\underset{n \rightarrow \infty}{\lim} \sum_{k=0}^{n-1} (-1)^k \lambda_k^{-2} (\sin(\lambda_k (wx + b + \frac{\pi}{2})) + \sin(\lambda_k(wx + b)))] \nonumber \\
&& \quad  [\underset{n \rightarrow \infty}{\lim} \sum_{j=0}^{n-1} (-1)^j \lambda_j^{-2} (\sin(\lambda_j (wx^\prime + b + \frac{\pi}{2})) + \sin(\lambda_j(wx^\prime + b)))] \, db \, dw
\end{eqnarray}
$$

Let us solve all the integrals with the case $$k \neq j$$

$$
\begin{eqnarray} \label{eq:prelucovariance} 
\int_{-\pi}^{\pi} &&p(b) \frac{(-1)^{j + k}}{\lambda_k^2 \lambda_j^2} (\sin(\lambda_k (wx + b + \frac{\pi}{2})) + \sin(\lambda_k(wx + b))) (\sin(\lambda_j (wx^\prime + b + \frac{\pi}{2})) + \sin(\lambda_j(wx^\prime + b))) \, db
\end{eqnarray} 
$$

Now, let us use the following trigonometry identity

$$\sin(\lambda_k(wx + b + \frac{\pi}{2})) = (-1)^{k} \cos(\lambda_k(wx + b))$$.

Plugging the above identity back to Equation \eqref{eq:prelucovariance}, we obtain

$$
\begin{eqnarray}
\frac{1}{2\pi} \frac{(-1)^{j + k}}{\lambda_k^2 \lambda_j^2} \int_{-\pi}^{\pi}  ((-1)^{k} \cos(\lambda_k(wx + b)) + \sin(\lambda_k(wx + b))) ((-1)^{j}\cos(\lambda_j(wx^\prime + b)) + \sin(\lambda_j(wx^\prime + b))) \, db  \nonumber \\
\end{eqnarray}
$$

Observe that we can extend the form above into 4 terms. For now, let us solve the integral for each term of the extended form above.

**Integral 1**

$$
\begin{eqnarray}
&& \int_{- \pi}^{\pi} (-1)^{j + k} \cos(\lambda_k(wx + b)) \cos(\lambda_j(wx^\prime + b)) \, db \nonumber \\
&&= \frac{1}{2} \int \cos(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j)) + \cos(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j)) db \nonumber \\
&&= \frac{1}{2} [\frac{\sin(w(\lambda_kx + \lambda_j x^\prime) + b(\lambda_k + \lambda_j))}{\lambda_k + \lambda_j} + \frac{\sin(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j))}{\lambda_k - \lambda_j}]_{-\pi}^{\pi} \nonumber \\
&&= 0 \nonumber
\end{eqnarray}
$$

We applied the identity $$\cos(x) \cos(y) = \frac{1}{2} (\cos(x + y) + \cos(x - y))$$ to obtain the second row. Note that $$\lambda_k, \lambda_j$$ are even, thus we have even shifting coefficient for each term. Since the lower bound and the upper bound are symmetric, we obtain $$0$$.

**Integral 2**

$$
\begin{eqnarray}
&& \int_{- \pi}^{\pi} \sin(\lambda_k(wx + b)) \sin(\lambda_j(wx^\prime + b)) \, db \nonumber \\
&&= \frac{1}{2} \int_{- \pi}^{\pi} \cos(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j)) - \cos(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j)) \, db \nonumber \\
&&= \frac{1}{2} [\frac{\sin(w(\lambda_k x - \lambda_j x^\prime) + b(\lambda_k - \lambda_j))}{\lambda_k - \lambda_j} - \frac{\sin(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j))}{\lambda_k + \lambda_j}] \nonumber \\
&&= 0 \nonumber
\end{eqnarray}
$$

We applied the identity $$\sin(x) \sin(y) = \frac{1}{2} (\cos(x - y) - \cos(x + y))$$ to obtain the second row. Due to the even shifting coefficient and symmetricity, we end up with $$0$$. 

**Integral 3**

$$
\begin{eqnarray}
&& \int_{- \pi}^{\pi} (-1)^k \cos(\lambda_k(wx + b)) \sin(\lambda_j(wx^\prime + b)) \, db  \nonumber \\
&&= \frac{(-1)^k}{2} \int_{- \pi}^{\pi}  \sin(w(\lambda_kx + \lambda_jx^\prime) + b(\lambda_k + \lambda_j)) + \sin(w(\lambda_jx^\prime - \lambda_kx) + b(\lambda_j - \lambda_k)) \nonumber \\
&&= \frac{(-1)^k}{2} [\frac{- \cos(w(\lambda_k x + \lambda_j x^\prime) + b(\lambda_k + \lambda_j))}{\lambda_k + \lambda_j} - \frac{\cos(w(\lambda_j x^\prime - \lambda_k x) + b(\lambda_j - \lambda_k))}{\lambda_j - \lambda_k}] \nonumber \\
&&= 0 \nonumber
\end{eqnarray}
$$

We applied the identity $$\sin(x) \cos(y) = \frac{1}{2} (\sin(x + y) + \sin(x - y))$$ to obtain the second row. Due to the even shifting coefficient and symmetricity, we end up with $$0$$. 

**Integral 4**

$$
\begin{eqnarray}
&& \int_{- \pi}^{\pi} (-1)^j \cos(\lambda_j(wx^\prime + b)) \sin(\lambda_k(wx + b)) = 0 \, db  \nonumber \\
\end{eqnarray}
$$

Following the exact steps in Integral 3, we obtain the same result for Integral 4. Now our integral only involves the term where $$j = k$$. Thus we can rewrite the covariance function as follows

$$
\begin{eqnarray}
k(x, x^\prime) &&= \int p(w) \int p(b) [\underset{n \rightarrow \infty}{\lim} \sum_{k=0}^{n-1} (-1)^k \lambda_k^{-2} (\sin(\lambda_k (wx + b + \frac{\pi}{2})) + \sin(\lambda_k(wx + b))) \nonumber \\
&& \quad (-1)^k \lambda_k^{-2} (\sin(\lambda_k (wx^\prime + b + \frac{\pi}{2})) + \sin(\lambda_k(wx^\prime + b)))] \, db \, dw
\end{eqnarray}
$$

By the dominant convergence theorem, we could plug the limit out of the integral and obtain

$$
\begin{eqnarray}
k(x, x^\prime) &&= \underset{n \rightarrow \infty}{\lim} \int p(w)  \sum_{k=0}^{n-1} (-1)^{2k} \lambda_k^{-4} \int p(b) (\sin(\lambda_k (wx + b + \frac{\pi}{2})) + \sin(\lambda_k(wx + b))) \nonumber \\
&& \quad (\sin(\lambda_k (wx^\prime + b + \frac{\pi}{2})) + \sin(\lambda_k(wx^\prime + b))) \, db \, dw \label{eq:almostcovarianceprelu}
\end{eqnarray}
$$

Let us solve the inner integral

$$
\begin{eqnarray}
&&\int_{- \pi}^{\pi} p(b) ((-1)^{k} \cos(\lambda_k (wx + b)) + \sin(\lambda_k(wx + b))) ((-1)^k \cos(\lambda_k (wx^\prime + b)) + \sin(\lambda_k(wx^\prime + b))) \, db \nonumber \\
&&= \int_{- \pi}^{\pi} p(b) ((-1)^{2k} \cos(\lambda_k (wx + b)) \cos(\lambda_k (wx^\prime + b)) + (-1)^k \cos(\lambda_k (wx + b)) \sin(\lambda_k(wx^\prime + b)) \nonumber \\
&& \quad + (-1)^{k} \sin(\lambda_k(wx + b)) \cos(\lambda_k (wx^\prime + b)) + \sin(\lambda_k(wx + b)) \sin(\lambda_k(wx^\prime + b)))  \, db \nonumber \\
&&= \int_{- \pi}^{\pi} p(b) ((-1)^{2k} \cos(w\lambda_k(x - x^\prime)) + (-1)^k \sin(w\lambda_k(x + x^\prime) + 2b\lambda_k)) \, db \nonumber \\
&&= \frac{1}{2 \pi} [(-1)^{2k} \cos(w\lambda_k(x - x^\prime)) b - (-1)^k \cos(w\lambda_k(x + x^\prime) + 2b\lambda_k) / 2 \lambda_k]_{- \pi}^\pi  \nonumber \\
&&= (-1)^{2k} \cos(w \lambda_k(x - x^\prime)) \nonumber \\
&&= \cos(w \lambda_k(x - x^\prime))
\end{eqnarray} 
$$

we obtained the third row by applying the identitites $$\sin(x \pm y) = \sin(x) \cos(y) \pm \cos(x) \sin(y)$$ and $$\cos(x \pm y) = \cos(x) \cos(y) \mp \sin(x) \sin(y)$$. Since $$2k$$ is guaranted to be even, we have $$(-1)^{2k} = 1$$. Now we insert the obtained equation to Equation \eqref{eq:almostcovarianceprelu}

\begin{equation}
k(x, x^\prime) = \underset{n \rightarrow \infty}{\lim} \int p(w) \sum_{k=0}^{n-1} (-1)^{2k} \lambda_k^{-4} \cos(w \lambda_k(x - x^\prime)) \, dw
\end{equation}

We can approximate the equation above by choosing only the first term. Furthermore, applying Euler's formula on the first term will give us 

\begin{equation}
k(x, x^\prime) = \int p(w) \exp(iwr)  \, dw
\end{equation}

Where $$r = x - x^\prime$$




