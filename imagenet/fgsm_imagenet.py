# -*- coding: utf-8 -*-
"""
Contributors:
    - Louis Rémus
"""
import os
import sys
from torchvision.models.inception import inception_v3
from PIL import Image
import matplotlib.pyplot as plt
import glob

# Add src path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fgsm import run_targeted_attack, run_non_targeted_attack, image_to_tensor, tensor_to_image, draw_result, \
    random_noise, run_non_targeted_attack_v3, get_label, plot_heatmap_results


def load_image(img_path):
    """
    Loads a PNG image and convert it to a PyTorch tensor image
    Args:
        img_path ():

    Returns:
        PyTorch tensor image
    """
    image = image_to_tensor(Image.open(img_path).convert('RGB'))
    return image


def load_images(input_dir):
    """
    Generates images
    Args:
        input_dir ():

    Returns:

    """
    for image_path in glob.glob(os.path.join(input_dir, '*.png')):
        yield load_image(image_path)


def plot_bunch_results():
    """
    Plots and saves a bunch of results
    Returns:

    """
    for img in images_gen:
        # Non-targeted
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
        plt.savefig('out/non_targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close(fig)

        plt.plot(losses)
        plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
        plt.ylabel('loss')
        plt.xlabel('n_iterations')
        plt.savefig(
            'out/non_targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close()

        # Targeted
        adv_img, noise, losses = run_targeted_attack(image=img, label=823, model=model, **kwargs)
        fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
        plt.savefig('out/targeted/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close(fig)

        plt.plot(losses)
        plt.title('Cross Entropy Loss\norig_label={},adversarial_label={}'.format(orig_label, adversarial_label))
        plt.ylabel('loss')
        plt.xlabel('n_iterations')
        plt.savefig('out/targeted/loss_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
        plt.close()


def plot_transferrable_noise():
    # Test transferable noise
    # Add the noise to another image in a targeted attack
    img = next(images_gen)
    adv_img, noise, losses = run_targeted_attack(image=img, label=834, model=model, **kwargs)
    fig, orig_label, adversarial_label = draw_result(img, noise, adv_img, model=model)
    plt.savefig('out/innocent/orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)

    innocent_img = next(images_gen)
    corrupted_img = innocent_img + noise
    fig, orig_label, adversarial_label = draw_result(innocent_img, noise, corrupted_img, model=model)
    plt.savefig('out/innocent/innocent_orig_label={},adversarial_label={}.png'.format(orig_label, adversarial_label))
    plt.close(fig)


def plot_step_size():
    img = next(images_gen)

    step_sizes_ = [0.1, 0.05, 0.01, 0.005, 0.001]
    all_losses = []

    for step_size_ in step_sizes_:
        print('step_size = {}'.format(step_size_))
        adv_img, noise, losses = run_targeted_attack(step_size=step_size_, image=img, label=823, model=model)
        all_losses.append(losses)

    plt.figure()
    for i, loss in enumerate(all_losses):
        plt.plot(loss, label='step_size = {}'.format(step_sizes_[i]))
    plt.title('Cross entropy loss and gradient descent step')
    plt.ylabel('loss')
    plt.xlabel('n_iterations')
    plt.legend()
    plt.savefig('out/loss_with_different_alphas.png')
    plt.close()


def benchmark_with_random_noise():
    # Comparaison with random noise
    different_label = 0
    fooled = 0
    iter_count = 0
    n_images = 10
    for img in images_gen:
        iter_count += 1
        print(iter_count)
        if iter_count > n_images:
            break

        # Model prediction on the image
        model_label = get_label(img, model)

        # Model prediction on the noisy image
        adv_img, noise, losses = run_non_targeted_attack(image=img, model=model, **kwargs)
        adversarial_label = get_label(adv_img, model)

        # Model prediction with random noise
        adv_img, noise = random_noise(img, model, kwargs['eps'])
        random_adversarial_label = get_label(adv_img, model)

        if adversarial_label != model_label:
            different_label += 1

        if model_label != random_adversarial_label:
            fooled += 1

    print(different_label, fooled, iter_count)


if __name__ == '__main__':
    # Parameters
    kwargs = {
        'eps': 2 * 8 / 225.,
        'n_iterations': 1,
        'step_size': 0.01
    }

    # Model
    model = inception_v3(pretrained=True, transform_input=True)

    # Instantiate the model
    model.eval()

    # Experiments
    images_gen = load_images('images')
    plot_bunch_results()

    print('over')
