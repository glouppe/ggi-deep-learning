class: middle, center, title-slide

# Deep Learning

A bottom-up introduction to deep neural networks (Part 2)

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](g.louppe@uliege.be)

---

# Outline

Goal: a bottom-up construction of neural networks.

- Neural networks
- Convolutional networks

---

class: middle

# Neural networks

---

# Cooking recipe

- Get data (loads of them).
- Get good hardware.
- Define the neural network architecture.
    - A neural network is a composition of differentiable functions.
    - Stick to non-saturating activation function to avoid vanishing gradients.
    - Prefer deep over shallow architectures.
- Optimize with (variants of) stochastic gradient descent.
    - Evaluate gradients with automatic differentiation.

---

# Perceptron

The perceptron (Rosenblatt, 1957) is one of the first mathematical models for a **neuron**. It is defined as:

$$f(\mathbf{x}) = \begin{cases}
   1 &\text{if } \sum_i w_i x_i + b \geq 0  \\\\
   0 &\text{otherwise}
\end{cases}$$

- This model was originally motivated by biology, with $w_i$ being synaptic weights and $x_i$ and $f$ firing rates.
- This is a **cartoonesque** biological model.

---

class: middle

Let us define the **activation** function:

$$\sigma(x) = \begin{cases}
   1 &\text{if } x \geq 0  \\\\
   0 &\text{otherwise}
\end{cases}$$
.center[![](figures/lec2/activation-sign.png)]

Therefore, the perceptron classification rule can be rewritten as
$$f(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b).$$

---

# Linear discriminant analysis

Consider training data $(\mathbf{x}, y) \sim P(X,Y)$, with
- $\mathbf{x} \in \mathbb{R}^p$,
- $y \in \\\{0,1\\\}$.

Assume class populations are Gaussian, with same covariance matrix $\Sigma$ (homoscedasticity):

$$P(\mathbf{x}|y) = \frac{1}{\sqrt{(2\pi)^p |\Sigma|}} \exp \left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu}_y)^T \Sigma^{-1}(\mathbf{x} - \mathbf{\mu}_y) \right)$$

---

<br>

Using the Bayes' rule, we have:

$$\begin{aligned}
P(Y=1|\mathbf{x}) &= \frac{P(\mathbf{x}|Y=1) P(Y=1)}{P(\mathbf{x})} \\\\
         &= \frac{P(\mathbf{x}|Y=1) P(Y=1)}{P(\mathbf{x}|Y=0)P(Y=0) + P(\mathbf{x}|Y=1)P(Y=1)} \\\\
         &= \frac{1}{1 + \frac{P(\mathbf{x}|Y=0)P(Y=0)}{P(\mathbf{x}|Y=1)P(Y=1)}}.
\end{aligned}$$

--

It follows that with

$$\sigma(x) = \frac{1}{1 + \exp(-x)},$$

we get

$$P(Y=1|\mathbf{x}) = \sigma\left(\log \frac{P(\mathbf{x}|Y=1)}{P(\mathbf{x}|Y=0)} + \log \frac{P(Y=1)}{P(Y=0)}\right).$$

---

class: middle

Therefore,

$$\begin{aligned}
&P(Y=1|\mathbf{x}) \\\\
&= \sigma\left(\log \frac{P(\mathbf{x}|Y=1)}{P(\mathbf{x}|Y=0)} + \underbrace{\log \frac{P(Y=1)}{P(Y=0)}}\_{a}\right) \\\\
    &= \sigma\left(\log P(\mathbf{x}|Y=1) - \log P(\mathbf{x}|Y=0) + a\right) \\\\
    &= \sigma\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu}\_1)^T \Sigma^{-1}(\mathbf{x} - \mathbf{\mu}\_1) + \frac{1}{2}(\mathbf{x} - \mathbf{\mu}\_0)^T \Sigma^{-1}(\mathbf{x} - \mathbf{\mu}\_0) + a\right) \\\\
    &= \sigma\left(\underbrace{(\mu\_1-\mu\_0)^T \Sigma^{-1}}\_{\mathbf{w}^T}\mathbf{x} + \underbrace{\frac{1}{2}(\mu\_0^T \Sigma^{-1} \mu\_0 - \mu\_1^T \Sigma^{-1} \mu\_1) + a}\_{b} \right) \\\\
    &= \sigma\left(\mathbf{w}^T \mathbf{x} + b\right)
\end{aligned}$$

---

class: middle, center

.width-100[![](figures/lec2/lda1.png)]

---

count: false
class: middle, center

.width-100[![](figures/lec2/lda2.png)]

---

count: false
class: middle, center

.width-100[![](figures/lec2/lda3.png)]

---

class: middle

Note that the **sigmoid** function
$$\sigma(x) = \frac{1}{1 + \exp(-x)}$$
looks like a soft heavyside:

.center[![](figures/lec2/activation-sigmoid.png)]

Therefore, the overall model
$f(\mathbf{x};\mathbf{w},b) =  \sigma(\mathbf{w}^T \mathbf{x} + b)$
is very similar to the perceptron.

---

class: middle

In terms of **tensor operations**, the computational graph of $f$ can be represented as:

.center.width-70[![](figures/lec2/graphs/logistic-neuron.png)]

where
- white nodes correspond to inputs and outputs;
- red nodes correspond to model parameters;
- blue nodes correspond to intermediate operations, which themselves produce intermediate output values (not represented).

This unit is the *core component* all neural networks!

---

# Logistic regression

Same model $$P(Y=1|\mathbf{x}) = \sigma\left(\mathbf{w}^T \mathbf{x} + b\right)$$ as for linear discriminant analysis.

But,
- **ignore** model assumptions (Gaussian class populations, homoscedasticity);
- instead, find $\mathbf{w}, b$ that maximizes the likelihood of the data.

---

class: middle

We have,

$$\begin{aligned}
&\arg \max\_{\mathbf{w},b} P(\mathbf{d}|\mathbf{w},b) \\\\
&= \arg \max\_{\mathbf{w},b} \prod\_{\mathbf{x}\_i, y\_i \in \mathbf{d}} P(Y=y\_i|\mathbf{x}\_i, \mathbf{w},b) \\\\
&= \arg \max\_{\mathbf{w},b} \prod\_{\mathbf{x}\_i, y\_i \in \mathbf{d}} \sigma(\mathbf{w}^T \mathbf{x}\_i + b)^{y\_i}  (1-\sigma(\mathbf{w}^T \mathbf{x}\_i + b))^{1-y\_i}  \\\\
&= \arg \min\_{\mathbf{w},b} \underbrace{\sum\_{\mathbf{x}\_i, y\_i \in \mathbf{d}} -{y\_i} \log\sigma(\mathbf{w}^T \mathbf{x}\_i + b) - {(1-y\_i)} \log (1-\sigma(\mathbf{w}^T \mathbf{x}\_i + b))}\_{\mathcal{L}(\mathbf{w}, b) = \sum\_i \ell(y\_i, \hat{y}(\mathbf{x}\_i; \mathbf{w}, b))}
\end{aligned}$$

This loss is an instance of the **cross-entropy** $$H(p,q) = \mathbb{E}_p[-\log q]$$ for  $p=Y|\mathbf{x}\_i$ and $q=\hat{Y}|\mathbf{x}\_i$.

---

class: middle

When $Y$ takes values in $\\{-1,1\\}$, a similar derivation yields the **logistic loss** $$\mathcal{L}(\mathbf{w}, b) = -\sum_{\mathbf{x}\_i, y\_i \in \mathbf{d}} \log \sigma\left(y\_i (\mathbf{w}^T \mathbf{x}\_i + b))\right).$$

.center[![](figures/lec2/logistic_loss.png)]

- In general, the cross-entropy and the logistic losses do not admit a minimizer that can be expressed analytically in closed form.
- However, a minimizer can be found numerically, using a general minimization technique such as **gradient descent**.

---

# Gradient descent

Let $\mathcal{L}(\theta)$ denote a loss function defined over model parameters $\theta$ (e.g., $\mathbf{w}$ and $b$).

To minimize $\mathcal{L}(\theta)$, **gradient descent** uses local linear information to iteratively move towards a (local) minimum.

For $\theta\_0 \in \mathbb{R}^d$, a first-order approximation around $\theta\_0$ can be defined as
$$\hat{\mathcal{L}}(\theta\_0 + \epsilon) = \mathcal{L}(\theta\_0) + \epsilon^T\nabla\_\theta \mathcal{L}(\theta\_0) + \frac{1}{2\gamma}||\epsilon||^2.$$

---

class: middle

A minimizer of the approximation $\hat{\mathcal{L}}(\theta\_0 + \epsilon)$ is given for
$$\begin{aligned}
\nabla\_\epsilon \hat{\mathcal{L}}(\theta\_0 + \epsilon) &= 0 \\\\
 &= \nabla\_\theta \mathcal{L}(\theta\_0) + \frac{1}{\gamma} \epsilon,
\end{aligned}$$
which results in the best improvement for the step $\epsilon = -\gamma \nabla\_\theta \mathcal{L}(\theta\_0)$.

Therefore, model parameters can be updated iteratively using the update rule:
$$\theta\_{t+1} = \theta\_t -\gamma \nabla\_\theta \mathcal{L}(\theta\_t)$$

Notes:
- $\theta_0$ are the initial parameters of the model;
- $\gamma$ is the **learning rate**;
- both are critical for the convergence of the update rule.

---

class: center, middle

![](figures/lec2/gd-good-0.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-1.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-2.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-3.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-4.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-5.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-6.png)

Example 1: Convergence to a local minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-7.png)

Example 1: Convergence to a local minima

---

class: center, middle

![](figures/lec2/gd-good-right-0.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-1.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-2.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-3.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-4.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-5.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-6.png)

Example 2: Convergence to the global minima

---

count: false
class: center, middle

![](figures/lec2/gd-good-right-7.png)

Example 2: Convergence to the global minima

---

class: center, middle

![](figures/lec2/gd-bad-0.png)

Example 3: Divergence due to a too large learning rate

---

count: false
class: center, middle

![](figures/lec2/gd-bad-1.png)

Example 3: Divergence due to a too large learning rate

---

count: false
class: center, middle

![](figures/lec2/gd-bad-2.png)

Example 3: Divergence due to a too large learning rate

---

count: false
class: center, middle

![](figures/lec2/gd-bad-3.png)

Example 3: Divergence due to a too large learning rate

---

count: false
class: center, middle

![](figures/lec2/gd-bad-4.png)

Example 3: Divergence due to a too large learning rate

---

count: false
class: center, middle

![](figures/lec2/gd-bad-5.png)

Example 3: Divergence due to a too large learning rate

---

# Stochastic gradient descent

In the empirical risk minimization setup, $\mathcal{L}(\theta)$ and its gradient decompose as
$$\begin{aligned}
\mathcal{L}(\theta) &= \frac{1}{N} \sum\_{\mathbf{x}\_i, y\_i \in \mathbf{d}} \ell(y\_i, f(\mathbf{x}\_i; \theta)) \\\\
\nabla \mathcal{L}(\theta) &= \frac{1}{N} \sum\_{\mathbf{x}\_i, y\_i \in \mathbf{d}} \nabla \ell(y\_i, f(\mathbf{x}\_i; \theta)).
\end{aligned}$$
Therefore, in **batch** gradient descent the complexity of an update grows linearly with the size $N$ of the dataset.

More importantly, since the empirical risk is already an approximation of the expected risk, it should not be necessary to carry out the minimization with great accuracy.

---

<br><br>

Instead, **stochastic** gradient descent uses as update rule:
$$\theta\_{t+1} = \theta\_t - \gamma \nabla \ell(y\_{i(t+1)}, f(\mathbf{x}\_{i(t+1)}; \theta\_t))$$

- Iteration complexity is independent of $N$.
- The stochastic process $\\\{ \theta\_t | t=1, ... \\\}$ depends on the examples $i(t)$ picked randomly at each iteration.

--

<br>

.grid.center.italic[
.kol-1-2[.width-100[![](figures/lec2/bgd.png)]

Batch gradient descent]
.kol-1-2[.width-100[![](figures/lec2/sgd.png)]

Stochastic gradient descent
]
]

---

class: middle

When decomposing the excess error in terms of approximation, estimation and optimization errors,
stochastic algorithms yield the best generalization performance (in terms of **expected** risk) despite being
the worst optimization algorithms (in terms of *empirical risk*) (Bottou, 2011).

$$\begin{aligned}
&\mathbb{E}\left[ R(\tilde{f}\_\*^\mathbf{d}) - R(f\_B) \right] \\\\
&= \mathbb{E}\left[ R(f\_\*) - R(f\_B) \right] + \mathbb{E}\left[ R(f\_\*^\mathbf{d}) - R(f\_\*) \right] + \mathbb{E}\left[ R(\tilde{f}\_\*^\mathbf{d}) - R(f\_\*^\mathbf{d}) \right]  \\\\
&= \mathcal{E}\_\text{app} + \mathcal{E}\_\text{est} + \mathcal{E}\_\text{opt}
\end{aligned}$$

---

# Layers

So far we considered the logistic unit $h=\sigma\left(\mathbf{w}^T \mathbf{x} + b\right)$, where $h \in \mathbb{R}$, $\mathbf{x} \in \mathbb{R}^p$, $\mathbf{w} \in \mathbb{R}^p$ and $b \in \mathbb{R}$.

These units can be composed *in parallel* to form a **layer** with $q$ outputs:
$$\mathbf{h} = \sigma(\mathbf{W}^T \mathbf{x} + \mathbf{b})$$
where  $\mathbf{h} \in \mathbb{R}^q$, $\mathbf{x} \in \mathbb{R}^p$, $\mathbf{W} \in \mathbb{R}^{p\times q}$, $b \in \mathbb{R}^q$ and where $\sigma(\cdot)$ is upgraded to the element-wise sigmoid function.

<br>
.center.width-70[![](figures/lec2/graphs/layer.png)]


---

# Multi-layer perceptron

Similarly, layers can be composed *in series*, such that:
$$\begin{aligned}
\mathbf{h}\_0 &= \mathbf{x} \\\\
\mathbf{h}\_1 &= \sigma(\mathbf{W}\_1^T \mathbf{h}\_0 + \mathbf{b}\_1) \\\\
... \\\\
\mathbf{h}\_L &= \sigma(\mathbf{W}\_L^T \mathbf{h}\_{L-1} + \mathbf{b}\_L) \\\\
f(\mathbf{x}; \theta) &= \mathbf{h}\_L
\end{aligned}$$
where $\theta$ denotes the model parameters $\\{ \mathbf{W}\_k, \mathbf{b}\_k, ... | k=1, ..., L\\}$.

- This model is the **multi-layer perceptron**, also known as the fully connected feedforward network.
- Optionally, the last activation $\sigma$ can be skipped to produce unbounded output values $\hat{y} \in \mathbb{R}$.

---

class: middle, center

.width-100[![](figures/lec2/graphs/mlp.png)]

---

class: middle

To minimize $\mathcal{L}(\theta)$ with stochastic gradient descent, we need the gradient $\nabla_\theta \mathcal{\ell}(\theta_t)$.

Therefore, we require the evaluation of the (total) derivatives
$$\frac{\text{d} \ell}{\text{d} \mathbf{W}\_k},  \frac{\text{d} \mathcal{\ell}}{\text{d} \mathbf{b}\_k}$$
of the loss $\ell$ with respect to all model parameters $\mathbf{W}\_k$, $\mathbf{b}\_k$, for $k=1, ..., L$.

These derivatives can be evaluated automatically from the *computational graph* of $\ell$ using **automatic differentiation**.

---

# Automatic differentiation

Consider a 1-dimensional output composition $f \circ g$, such that
$$\begin{aligned}
y &= f(\mathbf{u}) \\\\
\mathbf{u} &= g(x) = (g\_1(x), ..., g\_m(x)).
\end{aligned}$$
The **chain rule** of total derivatives states that
$$\frac{\text{d} y}{\text{d} x} = \sum\_{k=1}^m \frac{\partial y}{\partial u\_k} \underbrace{\frac{\text{d} u\_k}{\text{d} x}}\_{\text{recursive case}}$$

- Since a neural network is a composition of differentiable functions, the total
derivatives of the loss can be evaluated by applying the chain rule
recursively over its computational graph.
- The implementation of this procedure is called (reverse) **automatic differentiation** (AD).
- AD is not numerical differentiation, nor symbolic differentiation.

---

As a guiding example, let us consider a simplified 2-layer MLP and the following loss function:
$$\begin{aligned}
f(\mathbf{x}; \mathbf{W}\_1, \mathbf{W}\_2) &= \sigma\left( \mathbf{W}\_2^T \sigma\left( \mathbf{W}\_1^T \mathbf{x} \right)\right) \\\\
\mathcal{\ell}(y, \hat{y}; \mathbf{W}\_1, \mathbf{W}\_2) &= \text{cross\\\_ent}(y, \hat{y}) + \lambda \left( ||\mathbf{W}_1||\_2 + ||\mathbf{W}\_2||\_2 \right)
\end{aligned}$$
for $\mathbf{x} \in \mathbb{R^p}$, $y \in \mathbb{R}$, $\mathbf{W}\_1 \in \mathbb{R}^{p \times q}$ and $\mathbf{W}\_2 \in \mathbb{R}^q$.

--

.width-100[![](figures/lec2/graphs/backprop1.png)]

---

The total derivative $\frac{\text{d} \ell}{\text{d} \mathbf{W}\_1}$ can be computed **backward**, by walking through all paths from $\ell$ to $\mathbf{W}\_1$ in the computational graph and accumulating the terms:
$$\begin{aligned}
\frac{\text{d} \ell}{\text{d} \mathbf{W}\_1} &= \frac{\partial \ell}{\partial u\_8}\frac{\text{d} u\_8}{\text{d} \mathbf{W}\_1} + \frac{\partial \ell}{\partial u\_4}\frac{\text{d} u\_4}{\text{d} \mathbf{W}\_1} \\\\
\frac{\text{d} u\_8}{\text{d} \mathbf{W}\_1} &= ...
\end{aligned}$$

.width-100[![](figures/lec2/graphs/backprop2.png)]

---

class: middle

- This algorithm is known as **reverse-mode automatic differentiation**, also called **backpropagation**.
- An equivalent procedure can be defined to evaluate the derivatives in *forward mode*, from inputs to outputs.

---

# Vanishing gradients

Training deep MLPs with many layers has for long (pre-2011) been very difficult due to the **vanishing gradient** problem.
- Small gradients slow down, and eventually block, stochastic gradient descent.
- This results in a limited capacity of learning.

.width-100[![](figures/lec2/vanishing-gradient.png)]

.center[Backpropagated gradients normalized histograms (Glorot and Bengio, 2010).<br> Gradients for layers far from the output vanish to zero. ]

---

class: middle

Consider a simplified 3-layer MLP, with $x, w\_1, w\_2, w\_3 \in\mathbb{R}$, such that
$$f(x; w\_1, w\_2, w\_3) = \sigma\left(w\_3\sigma\left( w\_2 \sigma\left( w\_1 x \right)\right)\right). $$

Under the hood, this would be evaluated as
$$\begin{aligned}
u\_1 &= w\_1 x \\\\
u\_2 &= \sigma(u\_1) \\\\
u\_3 &= w\_2 u\_2 \\\\
u\_4 &= \sigma(u\_3) \\\\
u\_5 &= w\_3 u\_4 \\\\
\hat{y} &= \sigma(u\_5)
\end{aligned}$$
and its derivative $\frac{\text{d}\hat{y}}{\text{d}w\_1}$ as
$$\begin{aligned}\frac{\text{d}\hat{y}}{\text{d}w\_1} &= \frac{\partial \hat{y}}{\partial u\_5} \frac{\partial u\_5}{\partial u\_4} \frac{\partial u\_4}{\partial u\_3} \frac{\partial u\_3}{\partial u\_2}\frac{\partial u\_2}{\partial u\_1}\frac{\partial u\_1}{\partial w\_1}\\\\
&= \frac{\partial \sigma(u\_5)}{\partial u\_5} w\_3 \frac{\partial \sigma(u\_3)}{\partial u\_3} w\_2 \frac{\partial \sigma(u\_1)}{\partial u\_1} x
\end{aligned}$$

---

class: middle

The derivative of the sigmoid activation function $\sigma$ is:

.center[![](figures/lec2/activation-grad-sigmoid.png)]

$$\frac{\text{d} \sigma}{\text{d} x}(x) = \sigma(x)(1-\sigma(x))$$

Notice that $0 \leq \frac{\text{d} \sigma}{\text{d} x}(x) \leq \frac{1}{4}$ for all $x$.

---

class: middle

Assume that weights $w\_1, w\_2, w\_3$ are initialized randomly from a Gaussian with zero-mean and  small variance, such that with high probability $-1 \leq w\_i \leq 1$.

Then,

$$\frac{\text{d}\hat{y}}{\text{d}w\_1} = \underbrace{\frac{\partial \sigma(u\_5)}{\partial u\_5}}\_{\leq \frac{1}{4}} \underbrace{w\_3}\_{\leq 1} \underbrace{\frac{\partial \sigma(u\_3)}{\partial u\_3}}\_{\leq \frac{1}{4}} \underbrace{w\_2}\_{\leq 1} \underbrace{\frac{\sigma(u\_1)}{\partial u\_1}}\_{\leq \frac{1}{4}} x$$

This implies that the gradient $\frac{\text{d}\hat{y}}{\text{d}w\_1}$ **exponentially** shrinks to zero as the number of layers in the network increases.

Hence the vanishing gradient problem.

- In general, bounded activation functions (sigmoid, tanh, etc) are prone to the vanishing gradient problem.
- Note the importance of a proper initialization scheme.

---

# Rectified linear units

Instead of the sigmoid activation function, modern neural networks
are for most based on **rectified linear units** (ReLU) (Glorot et al, 2011):

$$\text{ReLU}(x) = \max(0, x)$$

.center[![](figures/lec2/activation-relu.png)]

---

class: middle

Note that the derivative of the ReLU function is

$$\frac{\text{d}}{\text{d}x} \text{ReLU}(x) = \begin{cases}
   0 &\text{if } x \leq 0  \\\\
   1 &\text{otherwise}
\end{cases}$$
.center[![](figures/lec2/activation-grad-relu.png)]

For $x=0$, the derivative is undefined. In practice, it is set to zero.

---

class: middle

Therefore,

$$\frac{\text{d}\hat{y}}{\text{d}w\_1} = \underbrace{\frac{\partial \sigma(u\_5)}{\partial u\_5}}\_{= 1} w\_3 \underbrace{\frac{\partial \sigma(u\_3)}{\partial u\_3}}\_{= 1} w\_2 \underbrace{\frac{\partial \sigma(u\_1)}{\partial u\_1}}\_{= 1} x$$

This **solves** the vanishing gradient problem, even for deep networks! (provided proper initialization)

Note that:
- The ReLU unit dies when its input is negative, which might block gradient descent.
- This is actually a useful property to induce *sparsity*.
- This issue can also be solved using **leaky** ReLUs, defined as $$\text{LeakyReLU}(x) = \max(\alpha x, x)$$ for a small $\alpha \in \mathbb{R}^+$ (e.g., $\alpha=0.1$).

---

# Universal approximation

.bold[Theorem.] (Cybenko 1989; Hornik et al, 1991) Let $\sigma(\cdot)$ be a
bounded, non-constant continuous function. Let $I\_p$ denote the $p$-dimensional hypercube, and
$C(I\_p)$ denote the space of continuous functions on $I\_p$. Given any $f \in C(I\_p)$ and $\epsilon > 0$, there exists $q > 0$ and $v\_i, w\_i, b\_i, i=1, ..., q$ such that
$$F(x) = \sum\_{i \leq q} v\_i \sigma(w\_i^T x + b\_i)$$
satisfies
$$\sup\_{x \in I\_p} |f(x) - F(x)| < \epsilon.$$

- It guarantees that even a single hidden-layer network can represent any classification
  problem in which the boundary is locally linear (smooth);
- It does not inform about good/bad architectures, nor how they relate to the optimization procedure.
- The universal approximation theorem generalizes to any non-polynomial (possibly unbounded) activation function, including the ReLU (Leshno, 1993).

---

class: middle

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-0.png)]

???

R: explain how to fit the components.

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-1.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-2.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-3.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-4.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-5.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-6.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-7.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-8.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-9.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-10.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-11.png)]

---

class: middle
count: false

Consider the 1-layer MLP
$$f(x) = \sum w\_i \text{ReLU}(x + b_i).$$
This model can approximate any smooth 1D function, provided enough hidden units.

.center[![](figures/lec2/ua-12.png)]

---

# (Bayesian) Infinite networks

What if $q \to \infty$?

Consider the 1-layer MLP with a hidden layer of size $q$ and a bounded activation function $\sigma$:

$$\begin{aligned}
f(x) &= b + \sum\_{j=1}^q v\_j h\_j(x)\\\\
h\_j(x) &= \sigma\left(a\_j + \sum\_{i=1}^p u\_{i,j}x\_i\right)
\end{aligned}$$

Assume Gaussian priors $v\_j \sim \mathcal{N}(0, \sigma\_v^2)$, $b \sim \mathcal{N}(0, \sigma\_b^2)$, $u\_{i,j} \sim \mathcal{N}(0, \sigma\_u^2)$ and $a\_j \sim \mathcal{N}(0, \sigma\_a^2)$.

---

class: middle

For a fixed value $x^{(1)}$, let us consider the prior distribution of $f(x^{(1)})$ implied by
the prior distributions for the weights and biases.

We have
$$\mathbb{E}[v\_j h\_j(x^{(1)})] = \mathbb{E}[v\_j] \mathbb{E}[h\_j(x^{(1)})] = 0,$$
since $v\_j$ and $h\_j(x^{(1)})$ are statistically independent and $v\_j$ has zero mean by hypothesis.

The variance of the contribution of each hidden unit $h\_j$ is
$$\begin{aligned}
\mathbb{V}[v\_j h\_j(x^{(1)})] &= \mathbb{E}[(v\_j h\_j(x^{(1)}))^2] - \mathbb{E}[v\_j h\_j(x^{(1)})]^2 \\\\
&= \mathbb{E}[v\_j^2] \mathbb{E}[h\_j(x^{(1)})^2] \\\\
&= \sigma\_v^2 \mathbb{E}[h\_j(x^{(1)})^2],
\end{aligned}$$
which must be finite since $h\_j$ is bounded by its activation function.

We define $V(x^{(1)}) = \mathbb{E}[h\_j(x^{(1)})^2]$, and is the same for all $j$.

---

class: middle

By the Central Limit Theorem, as $q \to \infty$, the total contribution
of the hidden units, $\sum\_{j=1}^q v\_j h\_j(x)$, to the value of $f(x^{(1)})$ becomes a Gaussian with variance $q \sigma_v^2 V(x^{(1)})$.

The bias $b$ is also Gaussian, of variance $\sigma\_b^2$, so for large $q$, the prior
distribution $f(x^{(1)})$ is a Gaussian of variance $\sigma\_b^2 + q \sigma_v^2 V(x^{(1)})$.

Accordingly, for $\sigma\_v = \omega\_v q^{-\frac{1}{2}}$, for some fixed $\omega\_v$, the prior $f(x^{(1)})$ converges to a Gaussian of mean zero and variance $\sigma\_b^2 + \omega\_v^2 \sigma_v^2 V(x^{(1)})$ as $q \to \infty$.

For two or more fixed values $x^{(1)}, x^{(2)}, ...$, a similar argument shows that,
as $q \to \infty$, the joint distribution of the outputs converges to a multivariate Gaussian
with means of zero and covariances of
$$\begin{aligned}
\mathbb{E}[f(x^{(1)})f(x^{(2)})] &= \sigma\_b^2 + \sum\_{j=1}^q \sigma\_v^2 \mathbb{E}[h\_j(x^{(1)}) h\_j(x^{(2)})] \\\\
&= \sigma\_b^2 + \omega_v^2 C(x^{(1)}, x^{(2)})
\end{aligned}$$
where $C(x^{(1)}, x^{(2)}) = \mathbb{E}[h\_j(x^{(1)}) h\_j(x^{(2)})]$ and is the same for all $j$.

---

class: middle

This result states that for any set of fixed points $x^{(1)}, x^{(2)}, ...$,
the joint distribution of $f(x^{(1)}), f(x^{(2)}), ...$ is a multivariate
Gaussian.

Therefore,  the infinitely wide 1-layer MLP converges towards
a  **Gaussian process**.

<br>

.center.width-80[![](figures/lec2/in.png)]

.center[(Neal, 1995)]

---

# Effect of depth

.bold[Theorem] (Montúfar et al, 2014) A rectifier neural network with $p$ input units and $L$ hidden layers of width $q \geq p$ can compute functions that have $\Omega((\frac{q}{p})^{(L-1)p} q^p)$ linear regions.

- That is, the number of linear regions of deep models grows **exponentially** in $L$ and polynomially in $q$.
- Even for small values of $L$ and $q$, deep rectifier models are able to produce substantially more linear regions than shallow rectifier models.

<br>
.center.width-80[![](figures/lec2/folding.png)]

---

class: middle

# Convolutional networks

???

R: check https://github.com/m2dsupsdlclass/lectures-labs/blob/master/slides/04_conv_nets/index.html
R: check http://www.deeplearningindaba.com/uploads/1/0/2/6/102657286/indaba_convolutional.pdf
R: capsule networks to fix the issue of only detecting the presence of stuff. Capsules also detect their composition.
R: add neocognitron story

---

class: middle

.center.width-80[![](figures/hubel-wiesel.png)]

.center[(Hubel and Wiesel, 1962)]

???

Given the non-satisfactory results of the MLPs, researchers in ML turned to domain knowledge for inspiration.

- Nobel prize in 1981 in Medicine for their study of the visual system in cats.
- Simple cells: detect features (convolutional layer+activation) -> activate upon specific patterns, mostly oriented edges
- Complex cells: achieve position invariance (pooling) -> activate upon specific patterns, regardless of their position in their receptive field

---

class: middle, center

.width-80[![](figures/lenet.png)]

Convolutional network (LeCun et al, 1989)

---

# Motivation

Suppose we want to train a fully connected network that takes $200 \times 200$ RGB images as input.

.center.width-50[![](figures/lec3/fc-layer1.png)]

What are the problems with this first layer?
- Too many parameters: $200 \times 200 \times 3 \times 1000 = 120M$.
- What if the object in the image shifts a little?

---

class: middle

## Fully connected layer

.center.width-40[![](figures/lec3/fc.png)]

In a **fully connected layer**, each hidden unit $h\_j = \sigma(\mathbf{w}\_j^T \mathbf{x}+b\_j)$ is connected to the entire image.
- Looking for activations that depend on pixels that are spatially far away is supposedly a waste of time and resources.
- Long range correlations can be dealt with in the higher layers.


---

class: middle

## Locally connected layer

.center.width-40[![](figures/lec3/lc.png)]

In a **locally connected layer**, each hidden unit $h\_j$ is connected to only a patch of the image.
- Weights are specialized locally and functionally.
- Reduce the number of parameters.
- What if the object in the image shifts a little?

---

class: middle

## Convolutional layer

.center.width-40[![](figures/lec3/conv-layer.png)]

In a **convolutional layer**, each hidden unit $h\_j$ is connected
to only a patch of the image, and **share** its weights with the other units $h\_i$.
- Weights are specialized functionally, regardless of spatial location.
- Reduce the number of parameters.

---

# Convolution

For one-dimensional tensors, given an input vector $\mathbf{x} \in \mathbb{R}^W$ and a convolutional kernel $\mathbf{u} \in \mathbb{R}^w$,
the discrete **convolution** $\mathbf{x} \star \mathbf{u}$ is a vector of size $W - w + 1$ such that
$$\begin{aligned}
(\mathbf{x} \star \mathbf{u})\_i &= \sum\_{m=0}^{w-1} x\_{i+m} u\_m .
\end{aligned}
$$

Technically, $\star$ denotes the cross-correlation operator. However, most machine learning
libraries call it convolution.

---

class: middle

.center[![](figures/lec3/1d-conv.gif)]

.footnote[Credits: [EE559 Deep Learning](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/dlc-slides-4a-dag-autograd-conv.pdf) (Fleuret, 2018)]

---

class: middle

Convolutions generalize to multi-dimensional tensors:
- In its most usual form, a convolution takes as input a 3D tensor $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$, called the input feature map.
- A kernel $\mathbf{u} \in \mathbb{R}^{C \times h \times w}$ slides across the input feature map, along its height and width. The size $h \times w$ is called the receptive field.
- At each location,  the element-wise product between the kernel and the input elements it overlaps is computed and the results are summed up.

---

class: middle

- The final output $\mathbf{o}$ is a 2D tensor of size $(H-h+1) \times (W-w+1)$ called the output feature map and such that:
$$\begin{aligned}
\mathbf{o}\_{j,i} &= \mathbf{b}\_{j,i} + \sum\_{c=0}^{C-1} (\mathbf{x}\_c \star \mathbf{u}\_c)\_{j,i} = \mathbf{b}\_{j,i} + \sum\_{c=0}^{C-1}  \sum\_{n=0}^{h-1} \sum\_{m=0}^{w-1}  \mathbf{x}\_{c,j+n,i+m} \mathbf{u}\_{c,n,m}
\end{aligned}$$
where $\mathbf{u}$ and $\mathbf{b}$ are shared parameters to learn.
- $D$ convolutions can be applied in the same way to produce a $D \times (H-h+1) \times (W-w+1)$ feature map,
where $D$ is the depth.
---

class: middle

.center[![](figures/lec3/3d-conv.gif)]

.footnote[Credits: [EE559 Deep Learning](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/dlc-slides-4a-dag-autograd-conv.pdf) (Fleuret, 2018)]

---

class: middle

In addition to the depth $D$, the size of the output feature map can be controlled through the stride and with zero-padding.

- The stride $S$ specifies the size of the step the kernel slides with.
- Zero-padding specifies whether the input volume is pad with zeroes around the border.
  This makes it possible to produce an output volume of the same size as the input. A hyper-parameter $P$ controls the amount of padding.

---

class: middle

.grid.center[
.kol-1-2[![](figures/lec3/no_padding_no_strides.gif)]
.kol-1-2[![](figures/lec3/padding_strides.gif)]
]
.grid.center[
.kol-1-2[
No padding ($P=0$),<br> no strides ($S=1$).]
.kol-1-2[
Padding ($P=1$),<br> strides ($S=2$).]
]

---

# Equivariance

A function $f$ is **equivariant** to $g$ if $f(g(\mathbf{x})) = g(f(\mathbf{x}))$.
- Parameter sharing used in a convolutional layer causes the layer to be equivariant to translation.
- That is, if $g$ is any function that translates the input, the convolution function is equivariant to $g$.
    - E.g., if we move an object in the image, its representation will move the same amount in the output.
- This property is useful when we know some local function is useful everywhere (e.g., edge detectors).
- However, convolutions are not equivariant to other operations such as change in scale or rotation.

---

# Pooling

When the input volume is large, **pooling layers** can be used to reduce the input dimension while
preserving its global structure, in a way similar to a down-scaling operation.

Consider a pooling area of size $h \times w$ and a 3D input tensor $\mathbf{x} \in \mathbb{R}^{C\times(rh)\times(sw)}$.
- Max-pooling produces a tensor $\mathbf{o} \in \mathbb{R}^{C \times r \times s}$
such that
$$\mathbf{o}\_{c,j,i} = \max\_{n < h, m < w} \mathbf{x}_{c,rj+n,si+m}.$$
- Average pooling produces a tensor $\mathbf{o} \in \mathbb{R}^{C \times r \times s}$ such that
$$\mathbf{o}\_{c,j,i} = \frac{1}{hw} \sum\_{n=0}^{h-1} \sum\_{m=0}^{w-1} \mathbf{x}_{c,rj+n,si+m}.$$

Pooling is very similar in its formulation to convolution. Similarly, a pooling layer can be parameterized in terms of its receptive field, a stride $S$ and a zero-padding $P$.

---

class: middle

.center[![](figures/lec3/pooling.gif)]

.footnote[Credits: [EE559 Deep Learning](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/dlc-slides-4a-dag-autograd-conv.pdf) (Fleuret, 2018)]

---

# Invariance

A function $f$ is **invariant** to $g$ if $f(g(\mathbf{x})) = f(\mathbf{x})$.
- Pooling layers can be used for building inner activations that are (slightly) invariant to small translations of the input.
- Invariance to local translation is helpful if we care more about the presence of a pattern rather than its exact position.

---

# Architectures

A **convolutional network** can often be defined as a composition of convolutional layers ($\texttt{CONV}$), pooling layers ($\texttt{POOL}$), linear rectifiers ($\texttt{RELU}$) and fully connected layers ($\texttt{FC}$).

<br>
.center.width-90[![](figures/lec3/convnet-pattern.png)]

---

class: middle

The most common convolutional network architecture follows the pattern:

$$\texttt{INPUT} \to [[\texttt{CONV} \to \texttt{RELU}]\texttt{\*}N \to \texttt{POOL?}]\texttt{\*}M \to [\texttt{FC} \to \texttt{RELU}]\texttt{\*}K \to \texttt{FC}$$

where:
- $\texttt{\*}$ indicates repetition;
- $\texttt{POOL?}$ indicates an optional pooling layer;
- $N \geq 0$ (and usually $N \leq 3$), $M \geq 0$, $K \geq 0$ (and usually $K < 3$);
- the last fully connected layer holds the output (e.g., the class scores).

---

class: middle

Some common architectures for convolutional networks following this pattern include:
- $\texttt{INPUT} \to \texttt{FC}$, which implements a linear classifier ($N=M=K=0$).
- $\texttt{INPUT} \to [\texttt{FC} \to \texttt{RELU}]{\*K} \to \texttt{FC}$, which implements a $K$-layer MLP.
- $\texttt{INPUT} \to \texttt{CONV} \to \texttt{RELU} \to \texttt{FC}$.
- $\texttt{INPUT} \to [\texttt{CONV} \to \texttt{RELU} \to \texttt{POOL}]\texttt{\*2} \to \texttt{FC} \to \texttt{RELU} \to \texttt{FC}$.
- $\texttt{INPUT} \to [[\texttt{CONV} \to \texttt{RELU}]\texttt{\*2} \to \texttt{POOL}]\texttt{\*3} \to [\texttt{FC} \to \texttt{RELU}]\texttt{\*2} \to \texttt{FC}$.

Note that for the last architecture, two $\texttt{CONV}$ layers are stacked before every $\texttt{POOL}$ layer. This is generally a good idea for larger and deeper networks, because multiple stacked $\texttt{CONV}$  layers can develop more complex features of the input volume before the destructive pooling operation.

---

class: center, middle, black-slide

.width-100[![](figures/lec3/convnet.gif)]

---

class: center, middle, black-slide

<iframe width="640" height="480" src="https://www.youtube.com/embed/FwFduRA_L6Q?&loop=1&start=0" frameborder="0" volume="0" allowfullscreen></iframe>

(LeCun et al, 1993)

---

class: middle

## LeNet-5

.center.width-100[![](figures/lec3/lenet.png)]
.center[(LeCun et al, 1998)]

---

class: middle

## AlexNet

.center.width-100[![](figures/lec3/alexnet.png)]
.center[(Krizhevsky et al, 2012)]

---

class: middle

## GoogLeNet

.center.width-100[![](figures/lec3/googlenet.png)]
.center[(Szegedy et al, 2014)]

---

class: middle

## VGGNet

.center.width-70[![](figures/lec3/vgg16.png)]
.center[(Simonyan and Zisserman, 2014)]

---

class: middle

## ResNet

.center.width-100[![](figures/lec3/resnet.png)]
.center[(He et al, 2015)]

---

class: middle

.center.width-100[![](figures/lec3/imagenet.png)]

---

class: middle

## Size

| &nbsp;&nbsp;&nbsp;Model | Size&nbsp;&nbsp;&nbsp; | Top-1 Accuracy&nbsp;&nbsp;&nbsp; | Top-5 Accuracy&nbsp;&nbsp;&nbsp; | Parameters&nbsp;&nbsp;&nbsp; | Depth&nbsp;&nbsp;&nbsp; |
| :----- | ----: | --------------: | --------------: | ----------: | -----: |
| Xception | 88 MB | 0.790 | 0.945| 22,910,480 | 126 |
| VGG16 | 528 MB| 0.715 | 0.901 | 138,357,544 | 23
| VGG19 | 549 MB | 0.727 | 0.910 | 143,667,240 | 26
| ResNet50 | 99 MB | 0.759 | 0.929 | 25,636,712 | 168
| InceptionV3 | 92 MB | 0.788 | 0.944 | 23,851,784 | 159 |
| InceptionResNetV2 | 215 MB | 0.804 | 0.953 | 55,873,736 | 572 |
| MobileNet | 17 MB | 0.665 | 0.871 | 4,253,864 | 88
| DenseNet121 | 33 MB | 0.745 | 0.918 | 8,062,504 | 121
| DenseNet169 | 57 MB | 0.759 | 0.928 | 14,307,880 | 169
| DenseNet201 | 80 MB | 0.770 | 0.933 | 20,242,984 | 201

.footnote[Credits: [Keras documentation](https://keras.io/applications/)]

---

# What does a network see?

Convolutional networks can be inspected by looking for input images $\mathbf{x}$ that maximize the activation $\mathbf{h}\_{\ell,d}(\mathbf{x})$ of a chosen convolutional kernel $\mathbf{u}$ at layer $\ell$ and index $d$ in the layer filter bank.

Such images can be found by gradient ascent on the input space:
$$\begin{aligned}
\mathcal{L}\_{\ell,d}(\mathbf{x}) &= ||\mathbf{h}\_{\ell,d}(\mathbf{x})||\_2\\\\
\mathbf{x}\_0 &\sim U[0,1]^{C \times H \times W } \\\\
\mathbf{x}\_{t+1} &= \mathbf{x}\_t + \gamma \nabla\_{\mathbf{x}} \mathcal{L}\_{\ell,d}(\mathbf{x}\_t)
\end{aligned}$$

---

class: middle

.width-100[![](figures/lec3/vgg16-conv1.jpg)]

.center[VGG16, convolutional layer 1-1, a few of the 64 filters]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

class: middle
count: false

.width-100[![](figures/lec3/vgg16-conv2.jpg)]

.center[VGG16, convolutional layer 2-1, a few of the 128 filters]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

class: middle
count: false

.width-100[![](figures/lec3/vgg16-conv3.jpg)]

.center[VGG16, convolutional layer 3-1, a few of the 256 filters]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

class: middle
count: false

.width-100[![](figures/lec3/vgg16-conv4.jpg)]

.center[VGG16, convolutional layer 4-1, a few of the 512 filters]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

class: middle
count: false

.width-100[![](figures/lec3/vgg16-conv5.jpg)]

.center[VGG16, convolutional layer 5-1, a few of the 512 filters]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

Some observations:
- The first layers appear to encode direction and color.
- The direction and color filters get combined into grid and spot textures.
- These textures gradually get combined into increasingly complex patterns.

In other words, the network appears to learn a hierarchical composition of patterns.

<br>
.width-70.center[![](figures/lec3/lecun-filters.png)]

---

What if we build images that maximize the activation of a chosen class output?

<br><br>

.center[
.width-40[![](figures/lec3/magpie.jpg)] .width-40[![](figures/lec3/magpie2.jpg)]
]

.center.italic[The left image is predicted with 99.9% confidence as a magpie.]

.footnote[Credits: [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)]

---

class: middle, center, black-slide

<iframe width="600" height="400" src="https://www.youtube.com/embed/SCE-QeDfXtA?&loop=1&start=0" frameborder="0" volume="0" allowfullscreen></iframe>

???

Start from an image $\mathbf{x}\_t$, offset the image by a random jitter, enhance some layer activation at multiple scales, zoom in, repeat on the produced image $\mathbf{x}\_{t+1}$.


---

class: end-slide, center
count: false

The end.
