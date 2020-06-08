"""Use adversarial perturbation to generate adversarial example"""

from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import sys

from step_2_pretrained import get_idx_to_label, get_image_transform, predict


def get_inputs():
    assert len(sys.argv) > 1, 'Need to pass path to image'
    image = Image.open(sys.argv[1])
    transform = get_image_transform()
    inputs = transform(image)[None]
    return inputs


def get_inverse_transform():
    return transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],  # INVERSE normalize images, according to https://pytorch.org/docs/stable/torchvision/models.html
        std=[1/0.229, 1/0.224, 1/0.255])


def tensor_to_image(tensor):
    x = tensor.data.numpy().transpose(1, 2, 0) * 255.
    x = np.clip(x, 0, 255)
    return Image.fromarray(x.astype(np.uint8))


def get_adversarial_example(inputs, r):
    x = inputs + r
    x = get_inverse_transform()(x[0])
    image = tensor_to_image(x)
    return image


def main():
    inputs = get_inputs()
    r = torch.Tensor(np.load('assets/adversarial_r.npy'))

    # save perturbed image
    os.makedirs('outputs', exist_ok=True)
    image = get_adversarial_example(inputs, r)
    image.save('outputs/adversarial.png')

    # check prediction is new class
    print(f'New prediction: {predict(image)}')


if __name__ == '__main__':
    main()
