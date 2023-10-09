import os
import sys
import argparse
import gc
import random
from pathlib import Path

from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor

from src.models.dift_sd import SDFeaturizer
from src.utils.visualization import Demo

import hloc
from hloc import extract_features, extractors
from hloc import match_features, matchers
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import read_image

sys.path.append(str(Path(hloc.__file__).parents[0] / '..' / 'third_party'))
from SuperGluePretrainedNetwork.models.superglue import (
    log_optimal_transport, arange_like
)


def get_feature_extractor(feature_type='superpoint_inloc'):
    conf = extract_features.confs[feature_type]
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval()
    return model, conf


def get_matcher(matcher_type='superglue'):
    conf = match_features.confs[matcher_type]
    conf['model']['weights'] = 'outdoor'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval()
    return model, conf

@torch.no_grad()
def run_feature_extractor(model, images, device):
    out = []
    for data in images:
        image_tensor = torch.from_numpy(data['image']).unsqueeze(0).to(
            device, non_blocking=True
        )

        pred = model({'image': image_tensor})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size = data['original_size']

        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
            if 'scales' in pred:
                pred['scales'] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()

        out.append(pred)

    return out


@torch.no_grad()
def run_superglue_matcher(matcher, keypoints):
    device = matcher.parameters().__next__().device

    data = {}
    for idx, pred in enumerate(keypoints):
        data[f'image{idx}'] = torch.empty((1,1)+tuple(pred['image_size'])[::-1])
        for key, value in pred.items():
            data[f'{key}{idx}'] = (
                torch.from_numpy(value).float().unsqueeze(0)
                .to(device, non_blocking=True)
            )

    pred = matcher(data)

    return pred


def read_image_hloc(filename, feat_conf, size):
    grayscale = feat_conf['preprocessing']['grayscale']
    resize_max = feat_conf['preprocessing']['resize_max']
    resize_force = feat_conf['preprocessing'].get('resize_force', False)
    interpolation = 'cv2_area'

    image = read_image(filename, grayscale=grayscale)
    image = image.astype(np.float32)
    image = extract_features.resize_image(image, size, interpolation)
    # size = image.shape[:2][::-1]

    if resize_max and (resize_force
                                 or max(size) > resize_max):
        scale = resize_max / max(size)
        size_new = tuple(int(round(x*scale)) for x in size)
        image = extract_features.resize_image(image, size_new, interpolation)

    if grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.

    data = {
        'image': image,
        'original_size': np.array(size),
    }
    return data


@torch.no_grad()
def get_matching_indices_wo_mutual(scores, match_threshold):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    zero = scores.new_tensor(0)
    mscores0 = max0.values # .exp()
    mscores1 = mscores0.gather(1, indices1)
    valid0 = (mscores0 > match_threshold)
    valid1 = valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return indices0, indices1, mscores0, mscores1


@torch.no_grad()
def get_matching_indices_with_mutual(scores, match_threshold):
    # Get the matches with score above "match_threshold".
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    # mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores0 = torch.where(mutual0, max0.values, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > match_threshold)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return indices0, indices1, mscores0, mscores1


@torch.no_grad()
def run_matcher_dift(
    keypoints, ft, bin_score, iters,
    match_threshold, sinkhorn=True, mutual=True
):
    assert len(keypoints) == 2
    assert np.all(keypoints[0]['image_size'] == keypoints[1]['image_size'])

    src_ft = ft[0].unsqueeze(0)
    src_ft = F.interpolate(
        src_ft, size=keypoints[0]['image_size'][::-1].tolist(), mode='bilinear'
    )
    x, y = keypoints[0]['keypoints'].T
    src_vec = src_ft[0, :, y, x] # (C, N)

    del src_ft
    gc.collect()
    torch.cuda.empty_cache()

    tgt_ft = ft[1].unsqueeze(0)
    tgt_ft = F.interpolate(
        tgt_ft, size=keypoints[1]['image_size'][::-1].tolist(), mode='bilinear'
    )
    x, y = keypoints[1]['keypoints'].T
    tgt_vec = tgt_ft[0, :, y, x] # (C, M)

    scores = torch.einsum('dn,dm->nm', src_vec, tgt_vec)

    # normalized cosine similarity
    norms = torch.einsum('n,m->nm', src_vec.norm(dim=0), tgt_vec.norm(dim=0))
    scores = scores / norms.clamp(min=1e-6)

    scores = scores.unsqueeze(0) # (1, N, M)

    del tgt_ft
    gc.collect()
    torch.cuda.empty_cache()

    if sinkhorn:
        descriptor_dim = src_vec.shape[0]
        scores *= (descriptor_dim ** 0.5)

        scores = log_optimal_transport(scores, bin_score, iters)
    else:
        # normalized cosine similarity
        # scores = (scores + 1) / 2 # [-1, 1] -> [0, 1]
        # scores.clamp_(min=1e-6)

        scores_ = torch.zeros(
            (1, scores.shape[1] + 1, scores.shape[2] + 1)
        ).to(scores.device)
        scores_[:, :-1, :-1] = scores
        scores = scores_

    get_matching_indices = (
        get_matching_indices_with_mutual if mutual
        else get_matching_indices_wo_mutual
    )

    indices0, indices1, mscores0, mscores1 \
        = get_matching_indices(scores, match_threshold)

    return {
        'matches0': indices0, # use -1 for invalid match
        'matches1': indices1, # use -1 for invalid match
        'matching_scores0': mscores0,
        'matching_scores1': mscores1,
    }


def keypoints_to_cv(keypoints):
    keypoints_cv = []

    for pred in keypoints:
        keypoints_cv.append([
            cv2.KeyPoint(i[0], i[1], size=1) for i in pred['keypoints']
        ])

    return keypoints_cv


def matches_to_cv(matches):
    matches_cv = [
        cv2.DMatch(i, match, _distance=1/(score+1e-6))
        for i, (match, score) in enumerate(zip(
            matches['matches0'].cpu().numpy().flatten().tolist(),
            matches['matching_scores0'].cpu().numpy().flatten().tolist(),
        ))
        if match >= 0
    ]

    return matches_cv


def sample_homography():
    H = np.eye(3)
    theta = random.uniform(-np.pi/8, np.pi/8)
    sx = random.uniform(0.08, 0.12)
    sy = random.uniform(0.08, 0.12)
    p1 = random.uniform(-0.01, 0.01)
    p2 = random.uniform(-0.01, 0.01)

    He = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    Ha = np.array([
        [1, sy, 0],
        [sx, 1, 0],
        [0, 0, 1],
    ])
    Hp = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [p1, p2, 1],
    ])
    H = He @ Ha # @ Hp
    return H


def homography_warp_image(filename):
    img0 = cv2.imread(filename)
    H = sample_homography()
    warped_img0 = cv2.warpPerspective(img0, H, img0.shape[:2][::-1])
    new_filename = Path('/tmp') / Path(filename).name.replace('.png', '_warped.png')
    cv2.imwrite(str(new_filename), warped_img0)
    return new_filename


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    category = 'cat' # random.choice(['cat', 'guitar'])

    if category == 'cat':
        filelist = ['./assets/cat.png', './assets/target_cat.png']
    elif category == 'guitar':
        filelist = ['./assets/guitar.png', './assets/target_guitar.png']

    warped_file = homography_warp_image(filelist[0])
    filelist[1] = warped_file

    prompt = f'a photo of a {category}'

    ft = []
    imglist = []
    hloc_images = []

    # decrease these two if you don't have enough RAM or GPU memory
    img_size = (616, 616) # 768
    ensemble_size = 5 # 8

    feat_extractor, feat_conf = get_feature_extractor()
    feat_extractor.to(device)

    matcher, matcher_conf = get_matcher()
    matcher.to(device)

    dift = SDFeaturizer()

    for filename in filelist:
        img = Image.open(filename).convert('RGB')
        img = img.resize(img_size)
        imglist.append(img)

        data = read_image_hloc(filename, feat_conf, img_size)
        hloc_images.append(data)

        img_tensor = PILToTensor()(img) / 255.0
        ft.append(dift.forward((img_tensor - 0.5) * 2,
                               prompt=prompt,
                               ensemble_size=ensemble_size))
    ft = torch.cat(ft, dim=0)

    keypoints = run_feature_extractor(feat_extractor, hloc_images, device)
    keypoints_cv = keypoints_to_cv(keypoints)

    matches_cv = {}

    matches = run_superglue_matcher(matcher, keypoints)
    matches_cv['superglue'] = matches_to_cv(matches)

    matches = run_matcher_dift(
        keypoints, ft,
        torch.ones_like(matcher.net.bin_score),
        matcher_conf['model']['sinkhorn_iterations'],
        # after sinkhorn the scores range is roughly [-8, 0]
        match_threshold=-0.58,
        sinkhorn=True, mutual=True,
    )
    matches_cv['dift_sinkhorn_womutual'] = matches_to_cv(matches)

    matches = run_matcher_dift(
        keypoints, ft, torch.ones_like(matcher.net.bin_score),
        matcher_conf['model']['sinkhorn_iterations'],
        # without sinkhorn the scores range is roughly [-1, 1]
        match_threshold=0.75,
        sinkhorn=False, mutual=True,
    )
    matches_cv['dift_wosinkhorn_womutual'] = matches_to_cv(matches)

    imgs_with_keypoints = []
    for img, pred in zip(imglist, keypoints_cv):
        img = np.array(img)

        img = cv2.drawKeypoints(img, pred, None, color=(255, 0, 0))

        # keypoints = pred['keypoints'].astype(int)
        # img[keypoints[:, 1], keypoints[:, 0], :] = [255, 0, 0]

        imgs_with_keypoints.append(img[..., ::-1])

    imgs_with_keypoints = cv2.hconcat(imgs_with_keypoints)
    cv2.imwrite('/tmp/keypoints.png', imgs_with_keypoints)

    for method, matches in matches_cv.items():
        imgs_with_matches = cv2.drawMatches(
            np.array(imglist[0]), keypoints_cv[0],
            np.array(imglist[1]), keypoints_cv[1],
            matches, None,
            # matchesThickness=3,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(f'/tmp/matches_{method}.png', imgs_with_matches)

    gc.collect()
    torch.cuda.empty_cache()
