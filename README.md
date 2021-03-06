[![Build Status](https://travis-ci.org/louis-r/adversarial_models.svg?branch=master)](https://travis-ci.org/louis-r/adversarial_models)

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

Let us say there is a classifier C and input sample X which we
call a clean example. Let us assume that sample X is correctly
classified by the classifier, i.e. C(X) = y_{true}. We can construct
an adversarial example A which is perceptually indistinguishable
from X but is classified incorrectly, i.e. C(A) != y_{true}.
These adversarial examples are misclassified far more often than
examples that have been perturbed by random noise, even if the magnitude
of the noise is much larger than the magnitude of the adversarial noise.

# Adversarial Attacks

The challenge is to make to design an adversarial noise so subtle that a
human observer does not even notice the modification, while the
classifier makes a mistake.

Note that the classifier is left untouched, which corresponds to the
real setting, where we rarely if ever have access to the classifier
design.

## Types of Adversarial Attacks

We distinguish are two types of adversarial attacks : the non-targeted
attacks and the targeted attacks.

1. **Non-targeted Attacks**  
The non-targeted attacks goal is to modify slightly the input (so
    the image) so that the unknown classifier misclassifies the input.
    In this case, we do not choose the classification of the fooled
    input.

2. **Targeted Attacks**  
Here, we choose the class of the fooled input. Indeed, given an
    image in class A that we want to misclassify in class B, we
    compute an adversarial noise to modify this image to have in
    classified in B by the unknown classifier.
    
# Results
## ImageNet - Inception
### Non Targeted Attacks

The following images presents the results of adversarial attacks
performed on the Inception model.


![Non targeted BIM attack on beetle](figures/non_targeted/orig_label=long-hornedbeetle,longicorn,longicornbeetle,adversarial_label=starfish,seastar.png)

![Non targeted BIM attack on cannon](figures/non_targeted/orig_label=cannon,adversarial_label=bassinet.png)

![Non targeted BIM attack on ostrich](figures/non_targeted/orig_label=ostrich,Struthiocamelus,adversarial_label=dhole,Cuonalpinus.png)


The following image (taken from Goodfellow CS231n 2017 lecture) shows
how the FGSM crosses the classification boundaries.

![Adversarial Map](figures/maps_adversarial.png)

### Targeted Attacks

We then perform targeted attacks against the same images, aiming for the
label stethoscope. Note that we reach the target label in all the cases
and we could have reached it with smaller number of iterations and thus
smaller distortion of the original image.

![Targeted BIM attack on beetle](figures/targeted/orig_label=long-hornedbeetle,longicorn,longicornbeetle,adversarial_label=stethoscope.png)

![Targeted BIM attack on cannon](figures/targeted/orig_label=cannon,adversarial_label=stethoscope.png)

![Targeted BIM attack on ostrich](figures/targeted/orig_label=bobsled,bobsleigh,bob,adversarial_label=stethoscope.png)


### Overall Performance

We performed a grid search over two hyperparameters, epsilon and
alpha. For each couple (epsilon, alpha), we compute the
percentage of successful attacks on 20 randomly selected images, with
1 iteration, therefore with a FGSM attack. The attack here is
non targeted.

![ImageNet: % of successful adversarial images on attacks](figures/non_targeted/imagenet_successful_adv_images.png)

# Conclusion

In this project, we explored the possibility of creating adversarial
images for two famous datasets: ImageNet and MNIST. We illustrated how
successful adversarial attacks were, leading us to think that the
classification boundary has a simple structure.

We also illustrated how the adversarial noise learned on one classifier, our was transferable to
another classifier, our , and conversely, which shows that the
adversarial noise generalizes well and has a common structure among
relatively similar classifiers. We also demonstrated that the structure
of the noise was crucial: targeted and non targeted were much more
effective than random noise attacks.

# Remarks

- It seems that every algorithm which is easy to optimize is easy to
    perturb: do we need to move past gradient-based optimization to
    overcome adversarial examples?

-   A smaller, but well-built noise for a non targeted attack has a
    better performance that a bigger random noise attack.

-   As we saw that adversarial noise was transferable, **we cannot use
    ensembles or bagging methods as a defense against adversarial
    attacks**.

-   One must consider the existence of adversarial examples when
    deciding whether to use machine learning.

-   We did not need access to the model parameters or the training set.
    This makes the task even easier for an attacker with bad intentions.

-   To actually assess the robustness of its model, a researcher must
    measure his model’s error rate on fast gradient sign method
    adversarial examples and report it.
