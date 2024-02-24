---
layout: post
title:  "Gradient Estimation with Measure Valued Derivatives"
---

### Introduction
In this post, I want to talk about stochastic gradient estimators and, along the way, introduce my master thesis. 
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
In all these cases we are estimating the gradient through sampling. The question is, how can we do this more efficiently? That is, with as few samples as possible while maintaining a good quality of the gradient. With quality I mean that the gradient should point in the direction of the true gradient, and that the variance of the gradient should be low. This is important because the quality of the gradient directly influences the convergence speed of the optimization algorithm (but also if it converges at all). To find a good gradient estimator, that was the goal of my master thesis.


### Shortcomings of the REINFORCE Gradient Estimator

---
[^1]: Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs, Ahmadian et al, [arxiv link](https://arxiv.org/pdf/2402.14740.pdf)
[^2]: Have a look at p.328 in [Barto's book](http://incompleteideas.net/book/RLbook2020.pdf) 
[^3]: Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Salimans et al, [arxiv link](https://arxiv.org/pdf/1703.03864.pdf)

