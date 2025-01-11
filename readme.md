# **Notes from [NYU Tandon ML 2022](https://chinmayhegde.github.io/fodl/)**

## **The three fundamental problems of machine learning**

- We have a dataset of input-label pairs $X = (x_i, y_i)_{i=1}^n \subset \mathbb{R}^d \times \mathbb{R}$, with $y \in \mathbb{R}$ and  $x \in \mathbb{R}^d$.
- This dataset is assumed to be acquired via *iid* sampling from a joint probability distribution $\mu(x, y)$, defined in space $\mathbb{R}^d \times \mathbb{R}$
- We want to find a prediction function $f$ that predicts label $y \in \mathbb{R}$ based on a previously unseen input $x \in \mathbb{R^d}$
- $f$ should be such that errors are small; at the population level, i.e., if we could sample the entirety of all possible $(x, y)$ pairs, **population risk** is defined as 
$$R(f) = \mathbb{E}_{(x,y) \sim \mu} l(y, f(x))$$ 
- $l$ is a loss function that compares predicted and observed labels.
- $\mathbb{E}_{(x,y) \sim \mu}$ highlights the fact that the expected value of the loss is computed with the probabilities given by $\mu(x,y)$ (the joint distribution of $x$ and $y$).
- In practice, we never know $\mu(x,y)$, so $R(f)$ cannot be computed. However, using training data, we can compute an 'alternative' version based on our sample: the **empirical risk**: 
$$\hat{R}(f) = \frac{1}{n} \sum_{i=1}^n l(y_i, f(x_i))$$
- The obvious next question is: which $f$? What architecture? How big? 
- Assuming we somehow pick a reasonable $f$ from a hypothesis class $\mathscr{F}$, now we have to optimize it, i.e., solve 
$$f_b = \arg \min_{f \in \mathscr{F}} \hat{R}(f)$$
- This optimization problem is never really solved in practice, i.e., we never really find $f_b$ (the 'best' model), all we do is reduce the empirical risk by changing our $f$'s weights and biases until some desired quality level is achieved. That optimized model is $\hat{f}$
- The expectation/hope in this whole process is that $\hat{f}$'s population risk $R(\hat{f})$ is small. Again, this cannot be done in practice because we only have samples, but at least theoretically we can say that $R(\hat{f})$ is made up of three components:

$$ R(\hat{f}) = [R(\hat{f}) - \hat{R}(\hat{f})] + [\hat{R}(f_b) - \hat{R}(\hat{f})] + [\hat{R}(f_b)]$$ 

- These three terms are:
    - 1st term: **generalization error**, arises because we're working with a sample and not the entire population, so we have a 'proxy' of the actual population risk
    - 2nd term: **optimization error**, arises because we're not really fully solving the optimization task of finding the actual minimum loss for our $f$, so we're not really finding $f_b$
    - 3rd term: **representation error**, arises because the best model $f_b$ will still have some error, and there may be other, better families (hypothesis classes $\mathscr{F}$) that lead to even smaller $l$

