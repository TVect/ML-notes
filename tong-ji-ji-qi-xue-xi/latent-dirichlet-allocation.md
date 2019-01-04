# 模型概述

![](/assets/Smoothed_LDA.png)

- **notation**

$$\alpha$$ is the parameter of the Dirichlet prior on the per-document topic distributions,
$$\beta$$ is the parameter of the Dirichlet prior on the per-topic word distribution,
$$\theta _{m}$$ is the topic distribution for document m,
$$ \varphi _{k} $$ is the word distribution for topic k,
$$ {\displaystyle z_{mn}} $$ is the topic for the n-th word in document m, and
$$ {\displaystyle w_{mn}} $$ is the specific word.

## **Generative process**

1. Choose $${\displaystyle \theta _{i}\sim \operatorname {Dir} (\alpha )}$$, where $$ i\in \{1,\dots ,M\}$$ and $$\mathrm {Dir} (\alpha )$$ is a Dirichlet distribution with a symmetric parameter $$\alpha$$  which typically is sparse $$\alpha < 1$$

2. Choose $${\displaystyle \varphi _{k}\sim \operatorname {Dir} (\beta )}$$, where $$ k\in \{1,\dots ,K\} $$ and $$\beta$$  typically is sparse

3. For each of the word positions $$ i,j $$, where $$ i\in \{1,\dots ,M\} $$, and $$ j\in \{1,\dots ,N_{i}\} $$

    (a) Choose a topic $${\displaystyle z_{i,j}\sim \operatorname {Multinomial} (\theta _{i}).}$$

    (b) Choose a word $${\displaystyle w_{i,j}\sim \operatorname {Multinomial} (\varphi _{z_{i,j}}).}$$



We can then mathematically describe the random variables as follows:

$$
{\displaystyle {\begin{aligned}{\boldsymbol {\varphi }}_{k=1\dots K}&\sim \operatorname {Dirichlet} _{V}({\boldsymbol {\beta }})\\{\boldsymbol {\theta }}_{d=1\dots M}&\sim \operatorname {Dirichlet} _{K}({\boldsymbol {\alpha }})\\z_{d=1\dots M,w=1\dots N_{d}}&\sim \operatorname {Categorical} _{K}({\boldsymbol {\theta }}_{d})\\w_{d=1\dots M,w=1\dots N_{d}}&\sim \operatorname {Categorical} _{V}({\boldsymbol {\varphi }}_{z_{dw}})\end{aligned}}}
$$



# 模型求解

模型中涉及到的参数有: $$\theta, \varphi, Z$$, 下面会用 Variational Inference 和 collapsed Gibbs sampling 两种方法进行求解.

## Variational Inference

the total probability of the model is:

$$
{\displaystyle P({\boldsymbol {W}},{\boldsymbol {Z}},{\boldsymbol {\theta }},{\boldsymbol {\varphi }};\alpha ,\beta )=\prod _{i=1}^{K}P(\varphi _{i};\beta )\prod _{j=1}^{M}P(\theta _{j};\alpha )\prod _{t=1}^{N}P(Z_{j,t}\mid \theta _{j})P(W_{j,t}\mid \varphi _{Z_{j,t}}),}
$$

现考虑用如下分解的变分分布: $$ q({\boldsymbol {Z}},{\boldsymbol {\theta }},{\boldsymbol {\varphi }}) = \prod \limits_{k=1}^{K}q(\varphi_k) \prod \limits_{i=1}^{M} \{q(\theta_i) \prod \limits_{j}^{N} q(z_{i,j}) \} $$ 对 $$P({\boldsymbol {Z}},{\boldsymbol {\theta }},{\boldsymbol {\varphi }} | {\boldsymbol {W}})$$ 做近似.

利用 mean field theory, 可以得到迭代公式.

- **求解 $$q(\theta_{i})$$**

$$
\begin{aligned}
q(\theta_{i}) & = Dirichlet(\alpha^*_1, \alpha^*_2, \dots, \alpha^*_K) \\
where \quad \alpha^*_k &= \alpha + \sum_{j} q(z_{i,j}=k)
\end{aligned}
$$

- **求解 $$q(\varphi_{k})$$**

$$ 
\begin{aligned}
q(\varphi_{k}) & = Dirichlet(\beta^*_1, \beta^*_2, \dots, \beta^*_V) \\
where \quad \beta^*_v & = \beta + \sum_{i,j} 1(w_{i,j}=v)q(z_{i,j}=k)
\end{aligned}
$$

- **求解 $$q(z_{i, j})$$**

$$ q(z_{i, j}=k) \propto exp(\psi(\alpha^*_k) - \psi(\sum_k \alpha^*_k) + \phi(\beta^*_{k, w_{i,j}}) - \phi(\sum_{v} \beta^*_{k, v}))$$


## collapsed Gibbs sampling


# 预测