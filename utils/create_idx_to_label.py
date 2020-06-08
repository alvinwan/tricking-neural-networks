"""
Downloads:
- https://github.com/Cadene/pretrained-models.pytorch/blob/master/data/imagenet_classes.txt
- https://github.com/Cadene/pretrained-models.pytorch/blob/master/data/imagenet_synsets.txt
"""

import json

with open('imagenet_classes.txt') as f:
    idx_to_synset = list(map(lambda s: s.strip(), f))


with open('imagenet_synsets.txt') as f:
    synset_to_name = {}
    for line in f:
        synset, name = line.split(' ', 1)
        synset_to_name[synset] = name.strip()

idx_to_name = {}
for idx, synset in enumerate(idx_to_synset):
    idx_to_name[idx] = synset_to_name[synset]

with open('imagenet_idx_to_label.json', 'w') as f:
    json.dump(idx_to_name, f)
