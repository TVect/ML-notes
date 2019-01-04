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

