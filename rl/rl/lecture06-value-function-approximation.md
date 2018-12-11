# Introduction

- Problem with large MDPs:

    - There are too many states and/or actions to store in memory
    
    - It is too slow to learn the value of each state individually

- Solution for large MDPs:

    - Estimate value function with function approximation

    $$\hat{v}(s; w) \approx  v_\pi(s)$$ , or  $$\hat{q}(s, a; w) \approx q_\pi(s, a)$$

    - Generalise from seen states to unseen states

    - Update parameter $$w$$ using MC or TD learning

# Incremental Methods

## Linear Value Function Approximation

Represent state by a feature vector: $$x(S) = (x_1(S), ..., x_n(S))^T$$.

Represent value function by a linear combination of features: $$ \hat{v}(S; w) = x(S)^T w = \sum_{j=1}^{n} x_j(S) w_j $$.

Objective function is quadratic in parameters $$w$$: $$ J(w) = E_{\pi} [(v_{\pi}(S) - x(S)^T w)^2]$$

Stochastic gradient descent converges on global optimum: 

$$ \triangledown_w \hat{v}(S; w) = x(S) $$

$$ \triangle w = \alpha (v_{\pi}(S) - \hat{v}(S; w)) x(S)$$

**Tips:**
特别地, 在有限状态的情况下, 取特征为 $$ x(S) = (1(S=s_1), 1(S=s_2), ..., 1(S=s_n))^T$$, 此时的linear value function approximation 即为 Table lookup.

## Incremental Prediction Algorithms

前面的做法中假设了已经知道了 true value function $$v_\pi(s)$$, 但这个 true value function 在实际的问题中是未知的.

In practice, we substitute a target for $$v_\pi(s)$$

- **For MC, the target is the return $$G_t$$**

    $$ \triangle w = \alpha (G_t − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w)$$

    Return Gt is an unbiased, noisy sample of true value $$v_\pi(S_t)$$
    
    Monte-Carlo evaluation converges to a local optimum, even when using non-linear value function approximation.

- **For TD(0), the target is the TD target $$R_{t+1 }+ \gamma \hat{v}(S_{t+1}; w)$$**

    $$ \triangle w = \alpha (R_{t+1 }+ \gamma \hat{v}(S_{t+1}; w) − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w) $$

    The TD(0)-target is a biased sample of true value $$v_\pi(S_t)$$

    Linear TD(0) converges (close) to global optimum.
    
- **For TD(λ), the target is the λ-return $$G_t^{\lambda}$$**

    The λ-return is also a biased sample of true value $$v_\pi(S_t)$$.

    - Forward view linear TD(λ):
        
        $$ \triangle w = \alpha (G_t^{\lambda} − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w) $$

    - Backward view linear TD(λ):
    
        $$ \delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}; w) − \hat{v}(St; w) $$
        
        $$ E_t = \gamma \lambda E_{t−1} + x(S_t) $$
        
        $$ \triangle w = \alpha \delta_t E_t $$


## Incremental Control Algorithms

# Batch Methods

