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

Objective function is quadratic in parameters $$w$$: $$ J(w) = E_{\pi} (v_{\pi}(S) - x(S)^T w) ^2$$

# Batch Methods

