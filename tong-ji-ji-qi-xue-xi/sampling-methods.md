# Sampling Methods

整理了一下机器学习中常用的采样方法。主要参考资料为 PRML 的第 11 章 Sampling Methods.

## Introduction

目的是要 evaluate $$E[f] = \int_{p(z)} f(z) dz = \int f(z) p(z) dz$$,

在离散变量的情形，积分号替换为求和号.

假定这个期望很复杂，不能通过解析的方法简单求解.

一般做法, 是从分布 $$p(z)$$ 中采样一些样本，用 $$\hat{f} = \frac{1}{L} \sum_{i=1}^{L} f(z_{i})$$ 去逼近 $$E(f)$$

问题在于，在采样过程中，样本可能并不是独立的

## Methods

### Basic Transformation Methods

假设可以由计算机得到 $$(0,1)$$ 均匀分布的随机数z，现在想得到服从任意分布p\(y\)的随机数y的样本，其中假定 $$y=f(z)$$。 概率密度变量转换时，为了保证概率相等，要求 $$p(z)dz~=p(y)dy$$，当然概率密度必须是正的，所以有下式：$$p(y)=p(z)|\frac{dz}{dy}|$$

为了能得到特定的分布，只要能找到相应的函数 $$f$$ 即可.

对上式两边求积分: $$z = \int_{-\inf}^{y} p(y) dy = h(y)$$, 取 $$f = h^{-1}$$, 也即 $$y$$ 的累积分布函数的逆，即可.

类似的，在多变量的情况下有：$$p(y_1, ..., y_m) = p(z_1, ..., z_m) | \frac{\partial (z_1, ..., z_m) }{\partial (y_1, ..., y_m)} |$$

例子: 指数分布，柯西分布，正态分布

这种方法需要对指定分布求无穷积分，只能适用于一些简单的分布。

### rejection sampling

假定现在要采样的分布是一个相对比较复杂的分布 $$p(z)$$, 很难直接采样，但是可以对给定的 $$z$$，求出未归一化的概率值 $$p(z) = \frac{1}{Z_p}\tilde{p}(z)$$

做拒绝采样的时候，需要一个 proposal distribution q\(z\), 这个分布相比之下，更简单，可以直接从中采样。另外有一个 contant $$k$$ 满足 $$k q(z) > \tilde{p}(z)$$。拒绝采样的流程如下:

1. 从分布 $$q(z)$$ 中采样一个 $$z_0$$.
2. 在 $$[0, kq(z_0)]$$ 中做均匀采样，得到一个数 $$u_0$$. 如果 $$u_0 > \tilde{p}(z_0)$$ 就拒绝样本, 反之接受样本 $$z_0$$.

要证明 rejection sampling 的有效性，只要证明 $$Q(z|accept) = \frac{q(z) Q(accept|z)}{Q(accept)} = p(z)$$ 即可. 这个等式是很显然的。

#### Adaptive rejection sampling

在基本的拒绝采样方法中，constant $$k$$ 的选取要很小心，$$k$$ 应该在满足 $$kq(z) > \tilde{p}(z)$$ 的约束下，要尽量小，以使得拒绝率不要太大。

所以，这就为 proposal distribution 的选取带来了难度，它应该尽量不要和待采样的分布差距太大。

在 $$p(z)$$ 是 log concave 的情况下，可以 adaptive 的构造一个 proposal distribution，进一步得到符合所需分布的样本。具体流程如下：

1. 取一个初始的网格点集合, 在集合中每个点处，构造一个 $$log p(x)$$的切线，这些切线连起来就构成了 $$log p(x)$$ 的一个 piece-wise linear upper bound $$u(x)$$.
2. 对 $$u(x)$$ 做 exp，再概率归一化. 即得到一个概率分布 $$s(x) = \frac{\exp u(x)}{\int \exp u(z) dz}$$. 这是一个 piece-wise exponential distribution。 **分段指数分布** 可以使用 Basic Transformation Methods 进行采样。
3. 把上面的分段指数分布当做 proposal distribution， 使用 rejection sampling 技术, 此时的 constant k 为 $$\int \exp u(z) dz$$。

   如果采样出来的点, 被拒绝了，就把该点再加到前面的网格点集合中，进一步得到新的 proposal distribution. 随着网格点的增多，proposal distribution 会和 $$p(x)$$ 越来越近似，拒绝率也会越来越低.

**弱点**

PRML Page-531 中，用正态分布的例子说明了在高维的情况下，即使 comparison function 和 required distribution 很接近，也会导出一个很低的接受率。

Furthermore, the exponential decrease of acceptance rate with dimensionality is a generic feature of rejection sampling.

所以，拒绝采样并不太适合于高维分布。

### importance sampling

importance sampling 主要用于估计指定分布下的期望，它本身没有提供从分布中采样的方法。

假定现在无法直接从分布 $$p(z)$$ 中采样，但是 $$p(z)$$ 的值可以方便的求得，要估计的期望为 $$E[f] = \int_{p(z)} f(z) dz = \int f(z) p(z) dz$$。

对这个问题，最简单直观的做法是将 z-space 离散化为一系列的网格点，将积分变为求和：$$E[f] \simeq \sum_{i=1}^{L} p(z^i)f(z^i)$$

但这种做法中的求和项会随着 z 的维度做指数增加。另外，使用均匀的网格点的时候，可能会导致大部分的求和项的的贡献值都很低，从而整个做法的效率也很低。理想上，会希望大部分的样本点的 $$p(z)$$ 比较大，或者更进一步 $$p(z)f(z)$$ 比较大

和前面的 rejection sampling 一样，做 importance sampling 的时候，也需要一个方便采样的 proposal distribution $$q(z)$$

$$
E_{p}[f] = \int f(z) p(z) dz = \int f(z) \frac{p(z)}{q(z)} q(z) dz = E_{q}[f(z) \frac{p(z)}{q(z)}] \simeq \frac {1}{L} \sum_{i=1}^{L} r_i f(z^i)
$$

其中, $$r_i = \frac{p(z^i)}{q(z^i)}$$ 表示的是重要性权重。

更一般的，在 p\(z\) 只能在差一个常数估计出来的情况下，importance sampling 也能使用。此时，$$p(z) = \frac{\tilde{p}(z)}{Z_p}$$, 其中 $$\tilde{p}(z)$$ 可以方便的计算出，而 $$Z_p$$ 是未知的。同样的，$$q(z) = \frac {\tilde{q}(x)}{Z_q}$$.

$$
E_{p}[f] = \int f(z) p(z) dz = \frac {Z_q}{Z_p} \int f(z) \frac{\tilde{p}(z)}{\tilde{q}(z)} q(z) dz \simeq \frac {Z_q}{Z_p} \frac {1}{L} \sum_{i=1}^{L} \tilde{r}_i f(z^i)
$$

其中，$$\tilde{r}_i = \frac{\tilde{p}(z)}{\tilde{q}(z)}$$

$$\frac{Z_p}{Z_q} = \frac{1}{Z_q} \int \tilde{p}(z) dz = \int \frac {\tilde{p}(z)} {Z_q} dz = \int \frac {\tilde{p}(z)} {\tilde{q}(z)}q(z) dz \simeq \frac{1}{L} \sum \tilde{r}_i$$

所以有：$$E[f] \simeq \sum_{i=1}^{L} w_i f(z^i)$$

其中, $$w_i = \frac {\tilde{r}_i}{\sum \tilde{r}_m} = \frac {\tilde{p}(z^i) / \tilde{q}(z^i)} {\sum \tilde{p}(z^m) / \tilde{q}(z^m)}$$

### Markov Chain Monto Carlo

#### Markov Chian

* **homogeneous**

  转移概率分布/转移概率矩阵与时间步无关

* **invariant / stationary distribution**

  一个特定变量的边缘概率可以表示为：$$p(z') = \sum q(z'|z)p(z)$$。

  对于一个概率分布 p 来说，如果 markov chain 中的每一步都让这个概率分布保持不变，那么说这个概率分布关于这个 markov chin 是 invariant 的.

* **detail balance**

  给定一个齐次的 markov chian，和一个 概率分布 p, 如果满足细致平稳性条件 $$p(z)q(z'|z) = p(z')q(z|z')$$，那么 p 即为该 markov chain 的一个 invariant distribution

* **ergodicity**

  对于任意给定的初始分布，都能收敛到一个指定的分布。这需要遍历性。

  在一个遍历的markov chain的情况下，把 invariant distribution 称为 equilibrium distribution。

  > a homogeneous Markov chain will be ergodic, subject only to weak restrictions on the invariant distribution and the transition probabilities

#### Metropolis-Hastings algorithm

选择一个概率分布 $$q(z^{t+1} | z^t)$$ 作为markov chain 的转移概率，再构造一个接受概率，以满足detail balance。

算法流程如下：

* 初始化 $$z_0$$
* 对于 i = 0, ..., N

  采样 $$u ∼ U(0; 1)$$

  采样 $$z' ∼ q(z | z_i)$$

  * if $$u < A(z', z_i) = \min (1, \frac{\tilde{p}(z')q(z_i|z')}{\tilde{p}(z_i)q(z'|z_i)} )$$ : $$z_{i+1} = z'$$
  * else: $$z_{i+1} = z_{i}$$

要证明 MH 的有效性，只要证明 detail balance 成立即可. 实际上有：

$$
\begin{aligned}
p(z)q(z'|z)A(z', z) & = p(z) q(z'|z) \min (1, \frac{\tilde{p}(z')q(z|z')}{\tilde{p}(z)q(z'|z_i)} )
\\\\
& = \min ( p(z)q(z'|z),  p(z')q(z|z') )
\\\\
& = p(z')q(z|z') \min (1, \frac{p(z)q(z'|z)}{p(z')q(z|z')})
\\\\
& = p(z')q(z|z')A(z, z')
\end{aligned}
$$

注意到，在 q 是对称的情况下，即简化为 Metropolis algorithm，此时每一步的接受率为 $$A(z', z) = \min (1, \frac{\tilde{p}(z')}{\tilde{p}(z)})$$

#### Gibbs Sampling

算法流程如下：

* 初始化 $$\{ z_i: i = 1, ..., M \}$$
* 对于 $$τ = 1, ..., T$$ :
  * 采样 $$z_1^{τ+1} ∼ p(z_1 | z_2^{τ}, z_3^{τ}, ..., z_M^{τ})$$
  * 采样 $$z_2^{τ+1} ∼ p(z_2 | z_1^{τ+1}, z_3^{τ}, ..., z_M^{τ})$$
  * . . .
  * 采样 $$z_j^{τ+1} ∼ p(z_j | z_1^{τ+1}, ..., z_{j-1}^{τ+1}, z_{j+1}^{τ}, ..., z_M^{τ})$$
  * . . .
  * 采样 $$z_M^{τ+1} ∼ p(z_M | z_1^{τ+1}, z_2^{τ+1}, ..., z_{M-1}^{τ+1})$$

Gibbs Sampling中有如下性质：

1. $$p(z)$$ is invariant distribution

   将每一步前后的分布记为 $$p(z), p({z}')$$. 假设该步骤是针对 $$z_k$$ 进行采样，可以知道采样前后的边缘分布 $$p(z_{-k}) = p(z_{-k}')$$, 因为采样前后 $$z_{-k} = z_{-k}'$$ 。另外，根据采样的转移概率，也显然有 $$p(z_k | z_{-k}) = p(z_k' | z_{-k}')$$. 所以，每一步前后的概率分布式不变的。

2. ergodicity

   遍历性的一个充分条件是没有条件概率分布处处为零。如果这个要求满足，那么z空间中的任意一点都可以从其他的任意一点经过有限步骤达到，这些步骤中每次对一个变量进行更新。

   如果这个要求没有满足，即某些条件概率分布为零，那么在这种情况下应用吉布斯采样时，必须显式地证明遍历性。

3. Gibbs Sampling 可以看做是 Metropolis-Hastings 算法的一个特殊情况

   考虑某个Metropolis-Hastings 采样的步骤，它涉及到变量 $$z_k$$，同时保持剩余的变量 $$z_{-k}$$不变，并且对于这种情形来说，从 $$z$$ 到 $$z'$$的转移概率为 $$q_k(z' | z) = p(z_k' | z_{-k})$$。此时有：

   $$
   A(z', z) = \frac {p(z')q_k(z|z')} {p(z)q_k(z'|z)} = \frac{p(z')p(z_k | z_{-k}')}{p(z)p(z_k'|z_{-k})} = \frac{p(z_{-k}')p(z_k'|z_{-k}')p(z_k | z_{-k}')}{p(z_{-k})p(z_k|z_{-k})p(z_k'|z_{-k})} = 1
   $$

   所以，Gibbs Sampling 可以被看做 Metropolis-Hastings算法接受率为1的一种特殊情况。

### Slice Sampling

为了从 $$p(x) = \frac{\hat{p}(x)}{Z_p}$$ 中采样\(其中仅可以 evaluate unnormalized distribution $$\hat{p}(x)$$ \), 引入一个新的变量 $$u$$, 构造一个联合分布 $$\hat{p}(x, u)$$:

![](../.gitbook/assets/slice.png)

容易知道，对上述联合分布求边缘分布有：

$$
\hat{p}(x) = \int_{0 \leq u \leq \hat{p}(x)} \frac{1}{Z_p} du = \frac{\hat{p}(x)}{Z_p} = p(x)
$$

$$\hat{p}(u) = \int_{\hat{p}(x) \geq u} \frac{1}{Z_p} dx$$

进一步有，$$\hat{p}(u|x)$$ 是 $$0 \leq u \leq \hat{p}(x)$$ 上的均匀分布， $$\hat{p}(x|u)$$ 是 $$\hat{p}(x) \geq u$$ 上的均匀分布。

总体来说，可以依次在 $$0 \leq u \leq \hat{p}(x)$$ 上的均匀采样 $$u$$, 在 $$\hat{p}(x) \geq u$$ 的 slice 上的均匀采样 $$x$$。最后忽略掉 $$u$$, 即得到服从 $$p(x)$$ 的分布样本。

## Reference

* PRML chapter 11. SAMPLING METHODS
* [有道云笔记上的版本](http://note.youdao.com/noteshare?id=4034e12c81b04fa16f0148f95169e522&sub=ACF2A85FBD9040518D68A63076C99D56)
* [博客版本](http://blog.tvect.cc/archives/318)

