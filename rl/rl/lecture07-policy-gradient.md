# Introduction

In this lecture we will directly parametrise the policy $$\pi_{\theta}(s, a) = P [a | s; θ]$$, instead of the state-value or action-value.

**Advantages & Disadvantages of Policy-Based RL**

- **Advantages**:

    Better convergence properties

    Effective in high-dimensional or continuous action spaces

    Can learn stochastic policies

- **Disadvantages**:

    Typically converge to a local rather than global optimum

    Evaluating a policy is typically inefficient and high variance

## Policy Search

Goal: given policy $$\pi_{\theta}(s, a)$$ with parameters $$\theta$$, find best $$\theta$$.

But how to measure the quality of a policy $$\pi_{\theta}$$?

- In episodic environments we can use the start value:

    $$ J_1(\theta) = V^{\pi_{\theta}}(s_1) = E_{\pi_{\theta}} [v_1] $$

- In continuing environments we can use the average value:
    
    $$ J_{avV} (\theta) = \sum_{s} d^{\pi_0}(s) V^{\pi_0}(s) $$
    
    Or the average reward per time-step:
    
    $$ J_{avR} (\theta) = \sum_{s} d^{\pi_0}(s) \sum_{a} \pi_{\theta}(s, a) R_s^a $$
    
    where $$d^{\pi_0}(s)$$ is stationary distribution of Markov chain for $$\pi_{\theta}$$


Policy based reinforcement learning is an optimisation problem, and we focus on gradient descent based methods.


# Finite Difference Policy Gradient

# Monte-Carlo Policy Gradient

# Actor-Critic Policy Gradient