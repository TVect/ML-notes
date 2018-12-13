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

### For MC, the target is the return $$G_t$$

$$ \triangle w = \alpha (G_t − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w)$$

Return Gt is an unbiased, noisy sample of true value $$v_\pi(S_t)$$
    
Monte-Carlo evaluation converges to a local optimum, even when using non-linear value function approximation.

### For TD(0), the target is the TD target $$R_{t+1 }+ \gamma \hat{v}(S_{t+1}; w)$$

$$ \triangle w = \alpha (R_{t+1 }+ \gamma \hat{v}(S_{t+1}; w) − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w) $$

The TD(0)-target is a biased sample of true value $$v_\pi(S_t)$$

Linear TD(0) converges (close) to global optimum.
    
### For TD(λ), the target is the λ-return $$G_t^{\lambda}$$

The λ-return is also a biased sample of true value $$v_\pi(S_t)$$.

- Forward view linear TD(λ):
        
    $$ \triangle w = \alpha (G_t^{\lambda} − \hat{v}(S_t; w)) \triangledown_w \hat{v}(S_t; w) $$

- Backward view linear TD(λ):
    
    $$ \delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}; w) − \hat{v}(St; w) $$
        
    $$ E_t = \gamma \lambda E_{t−1} + x(S_t) $$
        
    $$ \triangle w = \alpha \delta_t E_t $$


## Incremental Control Algorithms

**Policy evaluation**: Approximate policy evaluation, $$ \hat{q}(·, ·; w) \approx q_\pi $$

与前面的 Incremental Prediction Algorithms 类似, 只不过这里是对 $$ \hat{q}(·, ·; w) $$ 操作.

**Policy improvement**: $$\epsilon$$-greedy policy improvement

## Convergence

- **Convergence of Prediction Algorithms**

![](/assets/convergence-prediction.png)

- **Convergence of Control Algorithms**

![](/assets/convergence-control.png)


# Batch Methods

Gradient descent is simple and appealing, but it is not sample efficient.

Batch methods seek to find the best fitting value function given the agent’s experience ("training data").

## Least Squares Prediction

### **Stochastic Gradient Descent with Experience Replay**

Given experience consisting of < state; value > pairs

$$ D = \{ < s_1, v_1^\pi >, ..., < s_T, v_T^\pi > \}$$

Repeat:

1. Sample state, value from experience

    $$ < s, v^\pi > \sim D $$

2. Apply stochastic gradient descent update

    $$ \triangle w = \alpha (v^\pi − \hat{v}(S; w)) \triangledown_w \hat{v}(S; w)$$

Converges to least squares solution: $$ w^\pi = \arg \min LS(w) = \arg \min \sum (v_t^\pi − \hat{v}(S_t; w))^2$$


### **Experience Replay in Deep Q-Networks (DQN)**

DQN uses **experience replay** and **fixed Q-targets**

- Take action at according to $$\epsilon$$-greedy policy

- Store transition $$(s_t, a_t, r_{t+1}, s_{t+1})$$ in replay memory D

- Sample random mini-batch of transitions $$(s, a, r, s^\prime)$$ from D

- Compute Q-learning targets w.r.t. old, fixed parameters $$w^−$$

- Optimise MSE between Q-network and Q-learning targets

    $$L_i(w_i) = E_{s,a,r,s^\prime \sim D_i} [(r + \gamma \max_{a^\prime} Q(s^\prime, a^\prime; w_i^−) − Q(s, a; w_i)^2]$$

- Using variant of stochastic gradient descent

### **Linear Least Squares Prediction**

Experience replay finds least squares solution, but it may take many iterations.

Using linear value function approximation $$ \hat{v}(s; w) = x(s)^T w$$, We can solve the least squares solution directly.

求导数, 并令导数为0, 有:

$$ \alpha \sum_{i=1}^T (v_t^{\pi} - x(s_t)^T w) x(s_t) = 0 $$

$$ w = (\sum x(s_t) x(s_t)^T)^{-1} \sum x(s_t)v_t^\pi $$

For N features, direct solution time is $$O(N^3)$$. Incremental solution time is $$O(N^2)$$ using Shermann-Morrison.


In practice, We do not know true values $$v_t^\pi$$, and our "training data" must use noisy or biased samples of $$v_t^\pi$$.

- **LSMC** 

    Least Squares Monte-Carlo uses return $$v_t^\pi \approx G_t$$

- **LSTD** 
    
    Least Squares Temporal-Difference uses TD target $$v_t^\pi \approx R_{t+1} + \gamma \hat{v}(S_{t+1}; w)$$

- **LSTD(λ)** 

    Least Squares TD(λ) uses λ-return $$ v_t^\pi \approx G_t^\lambda $$


## Least Squares Control



















