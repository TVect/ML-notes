# Flow based generative models



这边整理一下 Flow-based generative models.

主要想法是要找一个性质良好变换 $$f$$, 将变量映射到一个隐空间, 使得变换之后的分布是一个简单的易于处理的分布. 通过这种操作可以得到精确的 log likelihood 的表达, 进而可以通过 MLE 做训练.

下面涉及到三篇文章, [NICE](https://arxiv.org/abs/1410.8516), [REAL NVP](https://arxiv.org/abs/1605.08803),[Glow](https://arxiv.org/abs/1807.03039) 都分别提出各种不同的变换形式 $$f$$, 以处理这个问题.

## NICE: non-linear independent components estimation

文章基于想法 `a good representation is one in which the data has a distribution that is easy to model`. 学习了一个非线性变换，将数据变换到隐空间上，使得变换之后的数据服从某种特定的简单的分布. 一方面这个变换可以比较容易的求 Jacobian 和求逆，另一方面可以通过这种变换的组合构造更加复杂的非线性变换. 训练准则是极大化 exact log-likelihood.

对于一个分布 $$p_X(x)$$_, 希望找到一个变换_ $$h = f(x)$$_, 将数据映射到一个新的空间, 在这个空间上的分布是可分解的:_ $$p_H(h) = \prod_{d} p_{H_d}(h_d)$$.

使用概率分布 basic transformation 的公式, 可知: $$p_X(x) = p_H(f(x)) |det \frac{\partial f(x)}{\partial x}|$$.

这篇 paper 中设计了变换$$f$$, 使其拥有性质: `"easy determinant of the Jacobian" and "easy inverse"`

### coupling layer

#### General coupling layer

$$
\begin{aligned}
x=(x_{I_1}, x_{I_2}) &\mapsto y=(y_{I_1}, y_{I_2}) \\
y_{I_1} &= x_{I_1} \\
y_{I_2} &= g(x_{I_2}; m(x_{I_1})) \\
\end{aligned}
$$

其中, $$g : R^{D−d} × m(R^d) \rightarrow R^{D−d}$$ 称为 `coupling law`, 其对于第一部分是一个可逆变换.

此时, 定义的变换的 Jacobian 为:

$$
\frac{\partial y}{\partial x} = \begin{pmatrix}
I_d & 0\\ 
\frac{\partial y_{I_2}}{\partial x_{I_1}} & \frac{\partial y_{I_2}}{\partial x_{I_2}}
\end{pmatrix}
$$

变换的逆变换为:

$$
\begin{aligned}
x_{I_1} &= y_{I_1} \\
x_{I_2} &= g^{-1}(y_{I_2}; m(y_{I_1})) \\
\end{aligned}
$$

#### Additive coupling layer

特别地，取上述 coupling law `g` 为: $$g\(x_{}; m\(x_{I\_1_}\)\) = x_{I\_2_} + m\(x_{I\_1}\)$$, 此时有:

$$
\begin{aligned}
y_{I_2} &= x_{I_2} + m(x_{I_1}) \\
x_{I_2} &= y_{I_2} - m(y_{I_1})) \\
\end{aligned}
$$

使用这种 additive coupling law 之后，计算变换的逆变换就很简单了, 而且此时的 Jacobian 即为一个单位阵. 另外 coupling function m 的选择没有什么限制, 可以是一个 domain & codomain 合适的神经网络.

#### Combining coupling layers

可以将多个 coupling 层堆叠在一起得到更加复杂的变换.

> Since a coupling layer leaves part of its input unchanged, we need to exchange the role of the two subsets in the partition in alternating layers, so that the composition of two coupling layers modifies every dimension.

特别地，对于Additive coupling layer，检查 combining 之后的 Jacobian，可以发现至少要 3 层才能保证每一维能影响到其他维度.

### allowing rescaling

> include a diagonal scaling matrix $S$ as the top layer, which multiplies the $i$-th ouput value by $S_{ii}$: $\(x\_i\)_{i \le D} \rightarrow \(S_{ii}x\_i\)_{i \le D}$. This allows the learner to give more weight \(i.e. model more variation\) on some dimensions and less in others.

此时, NICE criterion 形如: $$ \log(pX(x)) = \sum_{i=i}^D [\log(p{H_i}(f_i(x))) + \log(|S{ii}|)]$$

### prior distribution

we choose the prior distribution to be factorial, i.e.: $$p_H(h) = \prod_{i=1}^D p_{H_d}(h_d)$$.

其中的 $p\_{H\_d}\(h\_d\)$ 可以选为一些标准分布, 比如 gaussian distribution, logistic distribution.

## REAL NVP: real-valued non-volume preserving transformations

### Coupling layers

**affine coupling layer**

$$
\begin{aligned}
y_{1:d} &= x_{1:d} \\
y_{d+1:D} &= x_{d+1:D} \bigodot  \exp(s(x_{1:d})) + t(x_{1:d})
\end{aligned}
$$

where $$s$$ and $$t$$ stand for scale and translation, and are functions from $$R^d \rightarrow R^{D−d}$$, and $\bigodot$ is the Hadamard product or element-wise product.

The Jacobian of this transformation is:

$$
\frac{\partial y}{\partial x} = \begin{pmatrix}
 I_d & 0 \\ 
 \frac{\partial{y_{d+1: D}}}{\partial {x_{1:d}}} & diag(\exp[s(x_{1:d})]) 
\end{pmatrix}
$$

其行列式为: $$\exp [\sum_j s(x_{1:d})_j]$$

Since computing the Jacobian determinant of the coupling layer operation does not involve computing the Jacobian of $$s$$ or $$t$$, those functions can be arbitrarily complex. We will make them deep convolutional neural networks.

computing the inverse is no more complex than the forward propagation:

$$
\begin{aligned}
x_{1:d} &= y_{1:d} \\
x_{d+1:D} &= (y_{d+1:D} - t(x_{1:d})) \bigodot  \exp(-s(y_{1:d}))
\end{aligned}
$$

#### Masked convolution

Partitioning can be implemented using a binary mask b, and using the functional form for $$y$$:

$$y = b \bigodot x + (1-b) \bigodot (x \bigodot exp(s(b \bigodot x)) + t(b \bigodot x))$$

> We use two partitionings that exploit the local correlation structure of images: spatial checkerboard patterns, and channel-wise masking. The spatial checkerboard pattern mask has value 1 where the sum of spatial coordinates is odd, and 0 otherwise. The channel-wise mask b is 1 for the first half of the channel dimensions and 0 for the second half. For the models presented here, both s\(·\) and t\(·\) are rectified convolutional networks.

![](https://www.tvect.cn/wp-content/uploads/2019/10/mask_schema.jpg)

#### Combining coupling layers

Although coupling layers can be powerful, their forward transformation leaves some components unchanged. This difficulty can be overcome by composing coupling layers in an alternating pattern, such that the components that are left unchanged in one coupling layer are updated in the next

### Multi-scale architecture

![](https://www.tvect.cn/wp-content/uploads/2019/10/d08fc734415ef6bbc281bd4c90da28bc.png)

每个 $$f^{\(i\)}$$ 是一系列的 coupling-squeezing-coupling 操作.

对于 $$ f^{\(i \le L\)} $$ 有:

> we first apply three coupling layers with alternating checkerboard masks, then perform a squeezing operation, and finally apply three more coupling layers with alternating channel-wise masking.

对于 $$f^{\(L\)}$$ 有

> For the final scale, we only apply four coupling layers with alternating checkerboard masks.

**squeezing operation:** for each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s/2 × s/2 × 4c tensor

## Glow: Generative Flow with Invertible 1×1 Convolutions

### 常见 generative models 及 flow-based generative model 的优点

**generative models 简单分类:**

* **likelihood-based methods**
  1. Autoregressive models. Those have the advantage of simplicity, but have as disadvantage that synthesis has limited parallelizability, since the computational length of synthesis is proportional to the dimensionality of the data; this is especially troublesome for large images or video.
  2. Variational autoencoders \(VAEs\), which optimize a lower bound on the log-likelihood of the data. Variational autoencoders have the advantage of parallelizability of training and synthesis, but can be comparatively challenging to optimize.
  3. Flow-based generative models.
* **generative adversarial networks \(GANs\)**

**merits of flow-based generative models**

* Exact latent-variable inference and log-likelihood evaluation. In VAEs, one is able to infer only approximately the value of the latent variables that correspond to a datapoint. GAN’s have no encoder at all to infer the latents. In reversible generative models, this can be done exactly without approximation. Not only does this lead to accurate inference, it also enables optimization of the exact log-likelihood of the data, instead of a lower bound of it.
* Efficient inference and efficient synthesis. Autoregressive models, such as the Pixel- CNN, are also reversible, however synthesis from such models is difficult to parallelize, and typically inefficient on parallel hardware. Flow-based gener- ative models like Glow \(and RealNVP\) are efficient to parallelize for both inference and synthesis.
* Useful latent space for downstream tasks. The hidden layers of autoregressive models have unknown marginal distributions, making it much more difficult to perform valid manipulation of data. In GANs, datapoints can usually not be directly represented in a latent space, as they have no encoder and might not have full support over the data distribution. This is not the case for reversible generative models and VAEs, which allow for various applications such as interpolations between datapoints and meaningful modifications of existing datapoints.
* Significant potential for memory savings. Computing gradients in reversible neural networks requires an amount of memory that is constant instead of linear in their depth, as explained in the RevNet paper.

### Proposed Generative Flow

![](https://www.tvect.cn/wp-content/uploads/2019/10/c31a49d4f9032e76e77ac3e6932a0620.png)

![](https://www.tvect.cn/wp-content/uploads/2019/10/03460346b43806dd6b3c11921809f355.png)

#### Actnorm: scale and bias layer with data dependent initialization

> We propose an actnorm layer \(for activation normalizaton\), that performs an affine transformation of the activations using a scale and bias parameter per channel, similar to batch normalization. These parameters are initialized such that the post-actnorm activations per-channel have zero mean and unit variance given an initial minibatch of data. This is a form of data dependent initialization \(Salimans and Kingma, 2016\). After initialization, the scale and bias are treated as regular trainable parameters that are independent of the data.

#### Invertible 1 × 1 convolution

NICE & REAL NVP proposed a flow containing the equivalent of a permutation that reverses the ordering of the channels. We propose to replace this fixed permutation with a \(learned\) invertible 1 × 1 convolution, where the weight matrix is initialized as a random rotation matrix. Note that a 1 × 1 convolution with equal number of input and output channels is a generalization of a permutation operation.

The log-determinant of an invertible 1 × 1 convolution of a h × w × c tensor h with c × c weight matrix W is straightforward to compute:

$$
\log |det ( \frac{\partial conv2d(h; W)} {\partial h} )| = h · w · \log | det(W)|
$$

The cost of computing or differentiating $det\(W\)$ is $O\(c^3\)$, which is often comparable to the cost computing $conv2D\(h; W\)$ which is $O\(h · w · c^2\)$. We initialize the weights $W$ as a random rotation matrix, having a log-determinant of 0; after one SGD step these values start to diverge from 0.

**LU Decomposition**

This cost of computing $det\(W\)$ can be reduced from $O\(c^3\)$ to $O\(c\)$ by parameterizing $W$ directly in its LU decomposition: $W = PL\(U + diag\(s\)\)$

where $P$ is a permutation matrix, $L$ is a lower triangular matrix with ones on the diagonal, $U$ is an upper triangular matrix with zeros on the diagonal, and $s$ is a vector.

In this parameterization, we initialize the parameters by first sampling a random rotation matrix $W$, then computing the corresponding value of $P$ \(which remains fixed\) and the corresponding initial values of $L$ and $U$ and $s$ \(which are optimized\).

#### Affine Coupling Layers

类似于 Real NVP 中的 coupling layers，不过这里只保留了 channel-wise masking, 不再使用 spatial checkerboard patterns.

## 参考资料

* [NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION](https://arxiv.org/abs/1410.8516)
* [DENSITY ESTIMATION USING REAL NVP](https://arxiv.org/abs/1605.08803)
* [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039)
* [细水长flow之NICE：流模型的基本概念与实现](https://zhuanlan.zhihu.com/p/41912710)
* [RealNVP与Glow：流模型的传承与升华](https://zhuanlan.zhihu.com/p/43048337)

