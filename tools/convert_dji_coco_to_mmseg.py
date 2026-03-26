#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


NAME_TO_CLASS = {
    'cable': 1,
    'powelines': 1,
    'powerline': 1,
    'powerlines': 1,
    'tower_lattice': 2,
    'tower_tucohy': 3,
    'tower_wooden': 4,
    'void': 255,
}


def ann_to_binary_mask(segmentation, h, w):
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get('counts'), list):
        rle = mask_utils.frPyObjects(segmentation, h, w)
    else:
        rle = segmentation
    m = mask_utils.decode(rle)
    if m is None:
        return None
    if m.ndim == 3:
        m = np.any(m, axis=2)
    return m.astype(bool)


def convert_split(coco_root: Path, out_root: Path, split: str, out_split: str):
    ann_path = coco_root / split / '_annotations.coco.json'
    data = json.loads(ann_path.read_text())

    id_to_name = {c['id']: c['name'].strip().lower() for c in data.get('categories', [])}
    id_to_class = {cid: NAME_TO_CLASS[name] for cid, name in id_to_name.items() if name in NAME_TO_CLASS}

    img_dir = coco_root / split
    out_img_dir = out_root / 'leftImg8bit' / out_split
    out_mask_dir = out_root / 'gtFine' / out_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        anns_by_img[ann['image_id']].append(ann)

    converted = 0
    for img in data.get('images', []):
        src = img_dir / img['file_name']
        if not src.exists():
            continue

        stem = Path(img['file_name']).stem
        dst_img = out_img_dir / f'{stem}_leftImg8bit.png'
        dst_mask = out_mask_dir / f'{stem}_gtFine_labelTrainIds.png'

        rgb = Image.open(src).convert('RGB')
        rgb.save(dst_img)

        h, w = int(img['height']), int(img['width'])
        mask = np.zeros((h, w), dtype=np.uint8)

        anns = sorted(anns_by_img.get(img['id'], []), key=lambda a: float(a.get('area', 0)), reverse=True)
        for ann in anns:
            cls_val = id_to_class.get(ann.get('category_id'))
            if cls_val is None:
                continue
            bin_mask = ann_to_binary_mask(ann.get('segmentation'), h, w)
            if bin_mask is None:
                continue
            mask[bin_mask] = cls_val

        Image.fromarray(mask, mode='L').save(dst_mask)
        converted += 1

    print(f'{split} -> {out_split}: converted {converted} images')
    print('category mapping (COCO id -> train id):', id_to_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', required=True, type=Path)
    parser.add_argument('--out-root', required=True, type=Path)
    args = parser.parse_args()

    convert_split(args.coco_root, args.out_root, 'train', 'train')
    convert_split(args.coco_root, args.out_root, 'valid', 'val')


if __name__ == '__main__':
    main()
