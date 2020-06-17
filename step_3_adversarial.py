"""Use adversarial perturbation to generate adversarial example"""

from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import sys

from step_2_pretrained import get_idx_to_label, get_image_transform, predict, load_image


def get_inverse_transform():
    return transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],  # INVERSE normalize images, according to https://pytorch.org/docs/stable/torchvision/models.html
        std=[1/0.229, 1/0.224, 1/0.255])


def tensor_to_image(tensor):
    x = tensor.data.numpy().transpose(1, 2, 0) * 255.  # tensors are (channel x width x height). Transpose to get (height x width x channels)
    x = np.clip(x, 0, 255)  # ensure all image values are valid
    return Image.fromarray(x.astype(np.uint8))  # create image as array of unsigned integers


def get_adversarial_example(x, r):
    y = x + r
    y = get_inverse_transform()(y[0])
    image = tensor_to_image(y)
    return image


def main():
    inputs = load_image()
    r = torch.Tensor(np.load('assets/adversarial_r.npy'))

    # save perturbed image
    os.makedirs('outputs', exist_ok=True)
    adversarial = get_adversarial_example(inputs, r)
    adversarial.save('outputs/adversarial.png')

    # check prediction is new class
    print(f'Old prediction: {predict(inputs)}')
    print(f'New prediction: {predict(inputs + r)}')


if __name__ == '__main__':
    main()
