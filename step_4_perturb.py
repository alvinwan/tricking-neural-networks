"""Adversarial example generator for ImageNet-pretrained ResNet18"""
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import os

from step_2_pretrained import get_idx_to_label, get_image_transform, predict
from step_3_adversarial import get_adversarial_example, get_inputs


TARGET_LABEL = 1
EPSILON = 10 / 255.


def get_model():
    net = models.resnet18(pretrained=True).eval().float()
    r = nn.Parameter(data=torch.zeros(1, 3, 224, 224), requires_grad=True)
    return net, r


def main():
    print(f'Target class: {get_idx_to_label()[str(TARGET_LABEL)]}')
    net, r = get_model()
    inputs = get_inputs()
    labels = Variable(torch.Tensor([TARGET_LABEL])).long()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([r], lr=0.1, momentum=0.1)

    for i in range(30):
        r.data.clamp_(-EPSILON, EPSILON)
        optimizer.zero_grad()

        # forward + backward + optimize
        x = inputs + r
        outputs = net(x)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        print(f'Loss: {loss.item():.2f} / Class: {get_idx_to_label()[str(int(pred))]}')

    # save r
    np.save('outputs/adversarial_r.npy', r.data.numpy())

    # save perturbed image
    os.makedirs('outputs', exist_ok=True)
    image = get_adversarial_example(inputs, r)

    # check prediction is new class
    print(f'New prediction: {predict(image)}')

if __name__ == '__main__':
    main()
