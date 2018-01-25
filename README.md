# Introduction
A growing ensemble of tasks is being delegated to machine learning
algorithms. Among them, classification of images is a major subset,
especially with the recent developments of self-driving technology.
Recent studies by Google Brain have shown that any
machine learning classifier can be tricked to give incorrect
predictions.

Here, we successfully conducted targeted and non
targeted attacks on a pre-trained ImageNet Inception classifier and on a
custom neural classifier trained on the MNIST dataset, tuning attack
hyperparamaters to achieve the most efficient attacks. The success rate
of our attacks calls, in our opinion, for a need to move past
gradient-based optimization.

We also proved that the adversarial noise
learned on one model is transferable to another model, which means that
using ensemble or bagging methods as a defense against adversarial
examples will not work. We also proved that the adversarial noise
learned had a very particular structure related to the attacked image,
and could not be successfully applied to other images. Lastly, we
experimented several customized attacks on the ImageNet dataset.

# General Paradigm

Let us say there is a classifier +%24+C+%24+ and input sample $ X $ which we
call a clean example. Let us assume that sample $ X $ is correctly
classified by the classifier, i.e. $ C(X) = y_{true} $. We can construct
an adversarial example $ A $ which is perceptually indistinguishable
from $ X $ but is classified incorrectly, i.e. $ C(A) \neq y_{true} $.
These adversarial examples are misclassified far more often than
examples that have been perturbed by random noise, even if the magnitude
of the noise is much larger than the magnitude of the adversarial noise,
see (Szegedy et al., 2013 in [@DBLP:journals/corr/SzegedyZSBEGF13]).

\begin{align}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{align}