# Tensorflow implementation of a contractive Auto-Encoder [[1](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf)]

This is a personal attempt to reimplement a contractive autoencoder (with FC layers uniquely) as described in the original [paper](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf) by Rifai et Al.

To the best of our knowledge, this is the first implementation done with native Tensorflow.

I also provide in this repository extensions to the original contractive loss, to include a contraction term I computed for (in addition to the one provided in the [paper](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf) for 1 layer of sigmoid non linearity):
* 2 layers of sigmoid non linearity
* 1 layer of relu non linearity
* 2 layers of relu non linearity


The main files are:

- `contractive.py` provides the implementation of the auto-encoder with a loss of 2 terms: (1) Reconstruction, (2) Contraction

$$\mathcal{J} = \displaystyle\sum_{x \in D_n} \left(L(x, g(f(x)) + \lambda ||J_f(x)||^2_F \right)$$

        (f is the encoding function, g is the decoding function, L being the MSE in our case)
    


### Requirements:
The code requires tensorflow >= 2.0

### Implementation details and Jacobian calculations

$||J_f(x)||^2_F = \displaystyle\sum_{i, j} \left(\displaystyle\frac{\partial h_i(x)}{\partial x_j}\right)^2$

* $x$ is the input vector
* $h$ is the encoding vector


#### 1 layer of sigmoid activation:
$h = f(x) = sigmoid(Wx + b)$

with:
* $x = [x_1, x_2, ..., x_{dx}]^T$
* $h = [h_1, h_2, ..., h_{dh}]^T$
* $b = [b_1, b_2, ..., b_{dh}]^T$
* $d_h$: number of hidden dimensions of the encoding layer
* $d_x$: number of input dimensions
* W is of shape $[d_h, d_x]$


$\boxed{||J_f(x)||^2_F = \displaystyle\sum_{i=1}^{d_h} h_i^2 (1-h_i)^2 \displaystyle\sum_{j=1}^{d_x} w_{ij}^2}$


#### 2 layers of sigmoid activations:
$z(x) = sigmoid(W^{(1)}x + b^{(1)})$

$h = f(x) = f(z(x)) = sigmoid(W^{(2)}z + b^{(2)})$

with:
* $x = [x_1, x_2, ..., x_{dx}]^T$
* $z = [x_1, x_2, ..., x_{dz}]^T$
* $h = [h_1, h_2, ..., h_{dh}]^T$
* $b^{(1)} = [b^{(1)}_1, b^{(1)}_2, ..., b^{(1)}_{dz}]^T$
* $b^{(2)} = [b^{(2)}_1, b^{(2)}_2, ..., b^{(2)}_{dh}]^T$

* $d_h$: number of hidden dimensions of the encoding layer
* $d_z$: number of hidden dimensions of the first layer
* $d_x$: number of input dimensions
* $W^{(1)}$ is of shape $[d_z, d_x]$
* $W^{(2)}$ is of shape $[d_h, d_z]$




$\displaystyle\frac{\partial h_i(x)}{\partial x_j} = ?$


Applying the chain rule:
$\displaystyle\frac{\partial h_i(x)}{\partial x_j} = \displaystyle\sum_k \frac{\partial h_i}{\partial z_k} \frac{\partial z_k}{\partial x_j}$ 


$\displaystyle \frac{\partial h_i}{\partial z_k} = h_i (1 - h_i) w^{(2)}_{ik}$

$\displaystyle \frac{\partial z_k}{\partial x_j} = z_k (1 - z_k) w^{(1)}_{kj}$


$\displaystyle\frac{\partial h_i}{\partial x_j} = h_i (1 - h_i) \sum_k z_k (1 - z_k) w^{(1)}_{kj}w^{(2)}_{ik}$


$||J_f(x)||^2_F = \displaystyle\sum_{i, j} \left(\displaystyle\frac{\partial h_i(x)}{\partial x_j}\right)^2$

$\boxed{||J_f(x)||^2_F = \displaystyle\sum_{i=1}^{dh} h_i^2 (1-h_i)^2 \sum_{j=1}^{dx} \left(\sum_{k=1}^{dz} z_k (1 - z_k) w^{(1)}_{kj}w^{(2)}_{ik}\right)^2}$



#### 1 layer of ReLu activation:
$u(x) = Wx + b$ (preactivation)

$h = f(x) = relu(u(x)) = relu(Wx + b)$


$\displaystyle\frac{\partial h_i(x)}{\partial x_j} = w_{ij} \mathbb{1}_{u_i > 0} $ 

(I need to verify if the latter equation is correct when preactivation=0)


$\boxed{||J_f(x)||^2_F = \displaystyle\sum_{i=1}^{dh} \mathbb{1}_{u_i > 0} \sum_{j=1}^{dx} w_{ij}^2}$




#### 2 layers of ReLu activations:
$z(x) = relu(W^{(1)}x + b^{(1)})$

$h = f(x) = f(z(x)) = relu(W^{(2)}z + b^{(2)})$

Let's denote by $\tilde{h}$ the pre-activation of $h$ and $\tilde{z}$ the pre-activation of $z$

Applying again the chain rule:
$\displaystyle\frac{\partial h_i(x)}{\partial x_j} = \displaystyle\sum_k \frac{\partial h_i}{\partial z_k} \frac{\partial z_k}{\partial x_j}$ 


$\displaystyle \frac{\partial h_i}{\partial z_k} =  w^{(2)}_{ik} \mathbb{1}_{\tilde{h}_i > 0} $

$\displaystyle \frac{\partial z_k}{\partial x_j} =  w^{(1)}_{kj}  \mathbb{1}_{\tilde{z}_k > 0} $


$\displaystyle\frac{\partial h_i}{\partial x_j} = \mathbb{1}_{\tilde{h}_i > 0} \displaystyle\sum_{k=1}^{dz} \mathbb{1}_{\tilde{z}_k > 0} w^{(1)}_{kj} w^{(2)}_{ik} $ 


$\boxed{||J_f(x)||^2_F = \displaystyle\sum_{i=1}^{dh} \mathbb{1}_{\tilde{h}_i > 0} \sum_{j=1}^{dx} \left(\sum_{k=1}^{dz} \mathbb{1}_{\tilde{z}_k > 0} w^{(1)}_{kj}w^{(2)}_{ik}\right)^2}$


### Description of contractive plus:
`contractiveplus.py` provides a hybrid implementation of (1) a special auto-encoder from our prior [work](http://www.vldb.org/pvldb/vol12/p1934-zaouk.pdf) on modeling Spark Streaming workloads (2) the contractive auto-encoder. In this case, the loss function will have 3 terms: (1) Reconstruction term (2) Additional Input term (3) Contraction term

$$\mathcal{J} = \displaystyle\sum_{\{x, \theta\} \in D_n} \left(L(x, g(f(x)) + \gamma L(\theta, f_v(x)) + \lambda ||J_{f_{iv}}(x)||^2_F \right)$$

$f$ is the encoding function, $g$ is the decoding function, $L$ is the MSE in our case
        
Here we assume that the bottleneck layer $f(x)$ can be broken into two parts: 
* Invariant part: $f_{iv}(x)$ that's the desired encoding, and for which we add a Jacobian term
* Variant part: $f_v(x)$ that should approximate the known parameters $\theta$ that generated the observation $x$

So in this case, $f(x) = \left( f_{iv}(x) || f_{v}(x)\right )$

### Examples:
TODO


### References:
[[1](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf)] Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio. Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of
the 28th International Conference on International Conference on Machine Learning,
ICML’11, pages 833–840, USA, 2011. Omnipress.


<hr>
Last edied on April 17th, 2020 by K. Zaouk
