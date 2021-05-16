COCO_CLASSES_LIST = [
    'hand',
    'face',
    'mask'
]


def get_idx_to_class():
    return {i: n for i, n in enumerate(COCO_CLASSES_LIST)}
