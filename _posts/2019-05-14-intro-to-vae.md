---
categories:
  - Blog
title: Notes on VAEs
mathjax: true
---
{% include script.html %}
The authors of the original paper on VAEs first set out to approximate intractable posterior distributions by reparameterizing the variational lower bound to create an unbiased estimator for said lower bound.

## Problem assumptions
Let's assume the data arises from a random process involving a continuous latent variable $$z$$. From this, we have some prior distribution $$p_\theta(z)$$ and a value $$x^{(i)}$$ that is generated from the conditional distribution $$p_\theta(x|z)$$. These distributions must come from the same distribution family and have differentiable PDFs. To get these distributions, we need to solve the following three problems:
1. Efficiently estimate $$ \theta $$
2. Efficiently approximate the intractable posterior distribution $$p_\theta(z\mid x)$$
3. Efficient marginal inference (inferring the probability of one variable taking a particular value) of $$x$$
Given these three problems, let $$q_\phi(z|x)$$ represent the approximation of the intractable posterior distribution. We will jointly learn the parameters of the approximation ($$\phi$$) as well as the true distribution ($$\theta$$).

## Stochastic Gradient Variational Bayes (SGVB) estimator
We want to differentiate and optimize the lower bound $$\mathcal{L}(\theta, \phi;x^{(i)})$$ with respect to the variational parameters $$\phi$$ and generative parameters $$\theta$$. As mentioned before, we need to approximate the true (but intractable) posterior with some $$q_\phi(z|x)$$. We now reparameterize the latent variable using a differentiable transformation $$g$$ as follows: $$\tilde{z} = g_\phi(\epsilon, x)$$ where $$\epsilon ~ p(\epsilon)$$. By doing so, we make the previously indifferentiable random node $$z$$ differentiable, as the randomness is now carried in the $$\epsilon$$ node. We then use this deterministic $$z$$ in our calculation of the ELBO $$\mathcal{L}(\theta, \phi;x^{(i)})$$.