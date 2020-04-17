# Tensorflow implementation of a contractive Auto-Encoder [[1](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf)]

This is a personal attempt to reimplement a contractive autoencoder (with FC layers uniquely) as described in the original [paper](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf) by Rifai et Al.

To the best of our knowledge, this is the first implementation done with native Tensorflow.

I also provide in this repository extensions to the original contractive loss, to include a contraction term I computed for (in addition to the one provided in the [paper](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf) for 1 layer of sigmoid non linearity):
* 2 layers of sigmoid non linearity
* 1 layer of relu non linearity
* 2 layers of relu non linearity


More details <a href='README.ipynb'>here</a>.


### References:
[[1](https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf)] Salah Rifai, Pascal Vincent, Xavier Muller, Xavier Glorot, and Yoshua Bengio. Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of
the 28th International Conference on International Conference on Machine Learning,
ICML’11, pages 833–840, USA, 2011. Omnipress.


<hr>
Last edied on April 17th, 2020 by K. Zaouk
