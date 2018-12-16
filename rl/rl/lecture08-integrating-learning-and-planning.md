# Model-Based Reinforcement Learning

Learn a model from experience. & Plan value function (and/or policy) from model.

- Advantages:

Can efficiently learn model by supervised learning methods

Can reason about model uncertainty

- Disadvantages:

Two sources of approximation error. (First learn a model, then construct a value function)

## **What is a Model?**

A model M is a representation of an MDP $$ \left \langle S, A, P, R \right \rangle $$ parametrized by $$\eta$$.

We will assume state space $$S$$ and action space $$A$$ are known.

So a model $$M = \left \langle P, R \right \rangle $$ represents state transitions $$ P_\eta \approx P $$ and rewards $$R_\eta \approx R$$.

$$S_{t+1} \sim P_\eta(S_{t+1} | S_t, A_t)$$

$$R_{t+1} = R_\eta(R_{t+1} | S_t, A_t)$$

Typically assume conditional independence between state transitions and rewards:

$$ P[S_{t+1}, R_{t+1} | S_t, A_t] = P[S_{t+1} | S_t, A_t] P[R_{t+1} | S_t, A_t] $$

## **Learning a model**

eg. Table Lookup Model: 直接从 experience 中状态统计转移的频率和 reward 的平均.

Alternatively, 可以记下每个 experience tuple $$ \left \langle S_t, A_t, R_{t+1}, S_{t+1} \right \rangle $$, 在使用的时候随机选择一个与状态动作对匹配的 tuple.

## **Planning with a Model**

学习了 model 之后, 

- 可以直接用 Value iteration, Policy iteration, Tree search ... 做 planning. 

- 也可以先从 model 中 sample experience, 再使用 model free RL, 如 Monte-Carlo control, Sarsa, Q-learning...


# Integrated Architectures

**Dyna**:

- Learn a model from real experience

- Learn and plan value function (and/or policy) from real and simulated experience

**Dyna-Q Algorithm 算法**

![](/assets/dyna-q.png)

# Simulation-Based Search

## Monte Carlo Tree Search

![](/assets/mcts.jpg)

- **选择 Selection**：从根节点 R 开始，递归选择最优的子节点（后面会解释）直到达到叶子节点 L。

- **扩展 Expansion**：如果 L 不是一个终止节点（也就是，不会导致博弈游戏终止）那么就创建一个或者更多的字子节点，选择其中一个 C。

- **模拟 Simulation**：从 C 开始运行一个模拟的输出，直到博弈游戏结束。

- **反向传播** Backpropagation：用模拟的结果输出更新当前行动序列。

---

# 参考资料

- [mcts.ai](http://mcts.ai/) 包含有一些 mcts 的简单实现和例子.