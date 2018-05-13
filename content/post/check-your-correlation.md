---
title: Check your "correlation" matrix
date: 2018-05-12
tags: ["Python", "Statistics"]
---

Stochastic simulation is an essential tool for many businesses to play out likely and unlikely
scenarios. For example, in insurance it is used for capital modelling requirements. An [EU
directive](https://en.wikipedia.org/wiki/Solvency_II_Directive_2009) stipulates that an insurer
should hold enough capital to meet its obligations over a 12 month period at a 99.5% confidence
level (often quoted as the chance of an insurer being ruined during the year should be no more than
1 in 200).

Calculating the capital requirements is not trivial as this will depend on many variables with
different distributions so analytical solutions are almost always intractable which leads to the
need of stochastic simulation. What's more, the various variables will be correlated in intricate
ways. Before performing a capital modelling exercise, in addition to defining the parameters and the
distributions of the variables, the correlations between variables also have to be defined. It is
often the case that different people or different teams will provide the correlation information on
different subsets of variables and then at the end this is all brought together in one big
correlation matrix describing dependencies between all variables going into the model. But is the
resulting correlation matrix consistent?

Let's take a step back and remember what makes a correlation matrix. [A necessary and sufficient
condition](https://en.wikipedia.org/wiki/Covariance_matrix#Which_
matrices_are_covariance_matrices?) for a matrix to be a correlation matrix is that it is symmetric
and positive semi-definite.[^1] If \\(\Sigma\\) is our \\(n\times n\\) matrix, then we require

1. \\(\Sigma=\Sigma^\top\\) (symmetry)
2. \\(x^\top\Sigma x\geq 0\\) for all
vectors \\(x\in\mathbb{R}^n\\) (positive semi-definiteness).

[^1]: This is true for both correlation and covariance matrices.

OK, so if we have a matrix \\(\Sigma\\) we just need to check these two properties and we are good
to go with generating some scenarios. But how on earth can you verify positive semi-definiteness?
Luckily, an [equivalent condition](https://en.wikipedia.org/wiki/Positive-
definite_matrix#Characterizations) to this states that all eigenvalues of \\(\Sigma\\) must be non-negative.

Let's look at a toy example. Let

$$
\Sigma=
\left(\begin{matrix}
  1 & 0.75 & -0.75 \cr
  0.75 & 1 & 0 \cr
  -0.75 & 0 & 1
\end{matrix}\right),
$$

i.e. three variables, the first two highly positively correlated, the first and the third negatively
correlated and the second and the third independent. This matrix is clearly symmetric, but what
about its eigenvalues? In Python:

{{< highlight python3 >}}
import numpy as np
Σ=np.array([[1,0.75,-0.75],[0.75,1,0],[-0.75,0,1]])
np.linalg.eigvals(Σ)
{{</ highlight >}}
```
array([ 2.06066017, -0.06066017,  1.        ])

```

Uh oh, one of the eigenvalues is negative which makes it impossible for \\(\Sigma\\) to be a
correlation matrix. OK, but what happens if we try to simulate some random numbers anyway? Let's
simulate some multivariate normal random variables with mean \\(\mu=\mathbf{0}\\), standard
deviation \\(\sigma=\mathbf{1}\\) and covariance
\\(\text{diag}(\sigma)\Sigma\text{diag}(\sigma)=\Sigma\\):

{{< highlight python3 >}}
np.random.multivariate_normal([0,0,0],Σ)
{{</ highlight >}}
```
/usr/lib/python3.6/site-packages/ipykernel/__main__.py:1: RuntimeWarning: covariance is not positive-semidefinite.
  if __name__ == '__main__':
array([-0.05998612, -0.71491509, -0.36980048])
```

Numpy is kind enough to give us a warning that the supplied covariance matrix is not positive-definite,
but it will still generate random numbers---these will not have the required correlation structure
because it is impossible to realize! I would recommend running these checks yourself rather than
relying on a package to do it for you---some might not even give you a warning and you will be none
the wiser.

OK so our "correlation" matrix is no good, but the smallest eigenvalue is not that far from zero. Is
there a way to fix this? What we are looking for is a principled way to come up with a *bona fide*
correlation matrix that is as close as possible to the original matrix. This is an optimization
problem and will be the subject of an upcoming post.
