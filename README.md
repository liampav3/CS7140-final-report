# A Survey of Neural Network Architectures for Enhanced Predictive Uncertainty Estimation

## Abstract
Uncertainty awareness is a critical capability for artificial intelligence agents to ensure they make safe, rational decisions in applications where robustness matters. This survey covers two major lines of work in improving the predictive uncertainty estimation capabilities of deep learning architectures, Bayesian neural networks (BNN) and epistemic neural networks (ENN).

## Introduction
Uncertainty awareness is a critical prerequisite to many intelligent behaviors such as identifying avenues for further learning and accurately gauging one’s confidence in one’s decisions. Unfortunately, standard deep learning architectures tend to both underestimate their uncertainty and fail to identify its sources. This lack of uncertainty awareness often results in overconfidence in wrong, potentially harmful decisions \[1\]. As deep learning is posed to be deployed in safety-critical applications, like autonomous driving \[2\] and medical imaging \[3\], this deficiency becomes an increasingly pressing issue. Presented in this survey are two major lines of current work to improve the predictive uncertainty estimations of deep learning architectures: Bayesian neural networks (BNN) and epistemic neural networks (ENN). 

## Bayesian Neural Networks

BNNs consist of two components, a base neural architecture $f_w$ with parameters $w$ and a probability distribution over those parameters $q$. The model uses this probability distribution to make predictions and to quantify its uncertainty on those predictions. To make a prediction on some input $x$, a BNN computes the expectation 

$$\mathbb{E}_{w \sim q} f_w(x)$$


To generate the probability distribution over its parameters, a BNN begins with some predefined prior probability over its weights which is then conditioned on the training data form a posterior. Using exact Bayesian inference to compute the posterior is prohibitively expensive due to the high dimensionality of state-of-the-art neural architecture parameter spaces \[4\]. So, all BNNs compute some approximation of the posterior and BNN architectures primarily differ based on which approximation they employ. The difficulty of creating a BNN architecture lies in designing an approximation that balances fidelity to the true posterior and computational tractability. Two major types of approximation employed in current research are variational approaches \[5, 6\] and sample-based approaches \[7\].

### Variational Approaches

Variational inference seeks to model the true posterior over the model’s weights with a similar, but more computationally tractable distribution. This is achieved by defining a set of parameterized density functions and optimizing the parameters to minimize Kullbeck-Lieber divergence with respect to the posterior. The exact set of density functions and minimization procedure used is architecture-dependent (FIND A SOURCE). This survey covers two variational inference-based methods: Bayes by Backprop and MC-Dropout.

#### MC-Dropout
Dropout is a feature in which a network's inputs and/or hidden layer nodes are randomly set to 0 with some probability $p$. The MC-Dropout BNN \[5\] architecture adds and tunes dropout on its base network to simulate the network's behavior over the true posterior. Specifically, the random variable $W_i$ in the true posterior corresponding to the weights of the i-th layer is approximated by 

$$ W_i = M_i \cdot \text{diag}([z_{i,j}]_{j=1}^{K_i})$$

$$ z_{i,j} = \text{Bernoulli}(p_i)$$

where $L$ is the number of layers and $K_i$ is the dimensionaltiy of the $i$-th layer. This representation corresponds to some base weight matrix $M_i$ having its features randomly dropped out with probability $p_i$. The base matrices $M_i$ and the dropout probabilities $p_i$ are optimized to minimize an approximation of KL divergence with the true posterior.

Once the dropout approximation is fit, predictions and uncertainties are computed using the following Monte Carlo approximation of the expectation over the random variables $W_i$. $T$ independent samples of the random variables $W_i$ are drawn to form $T$ parameter sets $[\phi_1, ..., \phi_T]$. The prediction on some input $x$ is then computed as

$$\hat{f}(x) =  \frac{1}{T} \sum_{t=1}^{T} f_{\phi_t}(x)$$

which corresponds to averaging the models output over $T$ different runs with independently drawn dropout patterns. The uncertainty of this prediction is measure as

$$\text{Var}_{f(x)} = \tau^{-1}\mathbb{I} + \frac{1}{T} \sum_{t=1}^{T} \left(f_{\phi_t}(x)^T f_{\phi_t}(x)\right) - \hat{f}(x)^T \hat{f}(x)$$

where $\tau$ is a measure of model precision determined by the network's hyperparameters.



On an MNIST handwritten digit classification task, the MC-Dropout architecture was demonstrated to have appropriately uncertain (diffuse over multiple classes) softmax probability predictive distribution on ambiguous inputs. When used to extrapolate beyond its training data on a regression task, the MC-Dropout architecture had inaccurate predictive outputs but correctly returned that it was highly uncertain. A particularly interesting result is that the architecture reported an increasingly higher uncertainty as it extrapolated further past its training data (FIGURE).

However, another paper demonstrated that dropout methods may be approximating the risk of a model rather than its uncertainty \[7\]. The risk of a model is an inherent limit in how much variation in can capture. Uncertainty, on the other hand, is confusion over which model parameters are optimal. Consider the case of a model trying to predict the outcome of a weighted coin flip using only a predictive distribution over the sides. Even if the model perfectly predicts the weighted chance for each side, there is still a random chance that it guess wrong beyond its control. Uncertainty in this problem would be the model doubting that its predicted probability for each side is incorrect. Quanitfying risk is less useful for a learning model than uncerstainty because it cannot be resolved without a change in architecture \[7\].
#### Bayes by Backdrop

### Sample Approches

#### Deep Ensemble

## Epistemic Neural Networks


### Epinet

## Conclusion

## Bibliography

\[1\] Amodei, Dario, et al. "Concrete problems in AI safety." arXiv preprint arXiv:1606.06565 (2016).

\[2\] Grigorescu, Sorin, et al. "A survey of deep learning techniques for autonomous driving." Journal of Field Robotics 37.3 (2020): 362-386.

\[3\] Akay, Altug, and Henry Hess. "Deep learning: current and emerging applications in medicine and technology." IEEE journal of biomedical and health informatics 23.3 (2019): 906-920.

\[4\] Neal, Radford M. Bayesian learning for neural networks. Vol. 118. Springer Science & Business Media, 2012.

\[5\] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. PMLR, 2016.

\[6\] Blundell, Charles, et al. "Weight uncertainty in neural network." International conference on machine learning. PMLR, 2015.

\[7\] Osband, Ian. "Risk versus uncertainty in deep learning: Bayes, bootstrap and the dangers of dropout." NIPS workshop on bayesian deep learning. Vol. 192. 2016.
