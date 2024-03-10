---
layout: post
title:  "Gradient Estimation with Measure Valued Derivatives"
---

# 1. Introduction
In this post, I want to talk about stochastic gradient estimators and, along the way, introduce my [master thesis]({{ site.baseurl }}/assets/docs/thesis_pengfei.pdf). 
You probably have heard about *RLHF*, which is an application of Reinforcement Learning to align pretrained Large Language Models with human preferences (to a reward model modelling human preferences to be more specific).

Usually in RLHF, after obtaining the reward model you would train your LLM using PPO. Another valid approach, which was shown to improve on PPO[^1], is to use a very old algorithm from RL called the REINFORCE algorithm[^2].

$$
\mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ R(y, x) \nabla_\theta \log \pi_\theta(y|x) \right]
$$

Here \\( x \\) is the prompt, \\( y \\) is the continuation generated 
by the LLM \\( \pi_\theta(y|x) \\), whose parameters \\( \theta \\) 
we are trying to optimize. The equation above describes the gradient we use to maximize 
the reward in the form of an expectation. 
By doing a Monte-Carlo approximation, that is by sampling, we can obtain a **stochastic gradient estimate**. In theory, we only need one sample to obtain a gradient (This is also what they use in [^1]).

Actually the above formulation of the form of sampling from a distribution \\( p_\theta(x) \\) and evaluating its score \\( R(x) \\) is a very general formulation encountered widely, f.e. in episodic reinforcement learning, where \\( x \\) is a trajectory and \\( R(x) \\) is the sum of rewards along the trajectory.

$$
\mathbb{E}_{x \sim \pi_\theta(\cdot)} \left[ r(x) \right]
$$

Other instances include black-box optimization, where \\( x \\) is a point in the search space and \\( r(x) \\) is the value of the objective function at that point, or evolutionary algorithms, where \\( x \\) is a candidate solution and \\( r(x) \\) is the fitness of that solution[^3]. 
In all these cases we are estimating the gradient through sampling. The question is, how can we do this more efficiently? That is, with as few samples as possible while maintaining a good quality of the gradient. With quality I mean that the gradient should point in the direction of the true gradient (unbiased), and that the variance of the gradient should be low. This is important because the quality of the gradient directly influences the convergence speed of the optimization algorithm (but also if it converges at all). To find a good gradient estimator, that was the goal of my master thesis.


# 2.REINFORCE and Measure Valued Derivative Gradient Estimators
The **REINFORCE** gradient estimator, which we saw above, is easy to undestand, implement and unbiased, that is given enough samples it will converge to the true gradient. However, it has a high variance, which makes it slow to converge. This is because the gradient is multiplied by the reward, which can be very large or very small, leading to a high variance.

We can derive the REINFORCE estimator as follows 

$$
\nabla_{\theta} \mathbb{E}_{x \sim p_\theta(\cdot)} \left[ R(x) \right] = \int \nabla_{\theta} p_\theta(x) R(x) dx = \int \nabla_{\theta} p_\theta(x) \cdot \dfrac{p_\theta(x)}{p_\theta(x)}  R(x) dx  = 
\int \nabla_{\theta} \log p_\theta(x) p_\theta(x) R(x) dx = \mathbb{E}_{x \sim p_\theta(\cdot)} \left[ \nabla_{\theta} \log p_\theta(x) R(x) \right]
$$

where we used the identify \\( \dfrac{\nabla_{\theta} p_\theta(x)}{p_\theta(x)} = \nabla_{\theta} \log p_\theta(x) \\) and the interchangeability between integration and derrivative.

On the other hand, we have the **Measure-Valued Derivative** (MVD) Estimator. The MVD is based on a mathematical theorem that formulates the derivative of a distribution wrt its parameters as a difference between two distribtions. This formulation has been first introduced by Pflug in 1989 [^4], which went by relatively unnoticed in the machine learning community until it was picked up by Rosca et al in 2020 [^5] and
my supervisor Joao [^6]. Both have realized the potential of this underexplored gradient estimator and have shown that it leads to a lower variance than the Score Function Estimator (REINFORCE) for the cases of Bayesian Inference and Policy Gradients.

The MVD estimator is based on the following decomposition given by Pflug[^4]

$$
\nabla_{\theta} p_\theta(x) = c_\theta ( p_{\theta}^{+}(x) - p_{\theta}^{-}(x) )
$$

where each distribution has its own characteristic decomposition triplet \\( (c_\theta, p_{\theta}^{+}, p_{\theta}^{-}) \\). For a Gaussian distribution for example, the decomposition is given by

$$
 ( \frac{1}{\sigma \sqrt{2\pi}}, \theta + \sigma \mathcal{W}(2, 0.5), \theta - \sigma \mathcal{W}(2, 0.5) )
$$
    
where \\( \mathcal{W}(2, 0.5) \\) is a Weibull distribution with shape parameter 2 and scale parameter 0.5.
More decompositions can be found in my thesis or any other paper on MVDs.
The MVD estimator is then given by

$$
\nabla_{\theta} \mathbb{E}_{x \sim p_\theta(\cdot)} \left[ R(x) \right] = \mathbb{E}_{x \sim p_\theta^{+}(\cdot)} \left[ R(x) \right] - \mathbb{E}_{x \sim p_\theta^{-}(\cdot)} \left[ R(x) \right]
$$

Thus the MVD estimator is based on the difference between two expectations. It is unbiased and has a lower variance than the REINFORCE estimator. However, the big caveat here is that this decomposition and sampling has to be done for every dimension of the parameter space. For this reasons, I took a closer look at how we can utilize the MVD for improving the gradient quality in episodic reinforcement learning / black-box optimization / evolutionary algorithms.

# 3. Comparing REINFORCE and MVD
We can now compare both estimators in a simplified environment to check their gradient quality. Here we can use cost functions from the black-box optimization community such as the Rosenbrock function. 
The Rosenbrock function is a non-convex function that is used as a performance test problem for optimization algorithms. It is defined as 

$$
f(x, y) = (a - x)^2 + b(y - x^2)^2
$$

What we are looking for is a gradient that points in the direction of the true gradient and has a low variance using as few samples as possible.

Below we can see each gradient sample ploted as vector in a 2D parameter space. The true gradient is given by the green vector. The REINFORCE gradient is given by the pink vectors and the MVD gradient is given by the blue vectors. We can see that the MVD gradient points more in the direction of the true gradient and has a lesser variance in direction and magnitude than the REINFORCE gradient.

<div style="display: flex; justify-content: center;">
<img src="{{ site.baseurl }}/assets/images/reinforce_vs_mvd.png" alt="Your Image" style="width: 50%; height: 50%;" />
</div>

# 6. Combining REINFORCE and MVD

# 7. Conclusion
---
[^1]: Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs, Ahmadian et al, [arxiv link](https://arxiv.org/pdf/2402.14740.pdf)
[^2]: Have a look at p.328 in [Barto's book](http://incompleteideas.net/book/RLbook2020.pdf) 
[^3]: Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Salimans et al, [arxiv link](https://arxiv.org/pdf/1703.03864.pdf)
[^4]: [Pflug, G.C. Sampling derivatives of probabilities. Computing 42, 315–328 (1989). https://doi.org/10.1007/BF02243227](https://link.springer.com/article/10.1007/BF02243227)
[^5]: [Rosca et al, Measure-Valued Derivatives for Approximate Bayesian Inference](http:/bayesiandeeplearning.org/2019/papers/76.pdf)
[^6]: [Carvalho et al, An Empirical Analysis of Measure-Valued Derivatives for Policy Gradients](https://www.semanticscholar.org/reader/8f1eb8941f4a229a52bd122f4a8928922375e946)

