import random
import numpy as np
import sklearn
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision import transforms

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# resize and take the center part of image to what our model expects
def get_input_transform(normalize=True):
    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    else:
        normalize = lambda x: x
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transf

def get_preprocess_transform(normalize=True):
    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    else:
        normalize = lambda x: x
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return transf 

def get_input_tensors(img, normalize=True):
    transf = get_input_transform(normalize)
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])
    return transf

def batch_predict(images, model, item_tf):
    model.eval()
    batch = torch.stack(tuple(item_tf(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    return probs, logits

def mask_distance(masks, targets, features):
    dist = sklearn.metrics.pairwise_distances(
            masks,
            masks[0].reshape(1, -1),
            metric='cosine'
        ).ravel()
    return dist

def target_distance(masks, targets, features):
    dist = sklearn.metrics.pairwise_distances(
            targets,
            targets[0].reshape(1, -1),
            metric='cosine'
        ).ravel()
    return dist

def feature_distance(masks, targets, features):
    dist = sklearn.metrics.pairwise_distances(
            features,
            features[0].reshape(1, -1),
            metric='cosine'
        ).ravel()
    return dist

def exponential_kernel(d, width=.25):
    return np.sqrt(np.exp(-(d ** 2) / width ** 2))

def quickshift_segmentation(img):
    segments = quickshift(x,
        kernel_size=4,
        max_dist=200,
        ratio=0.2)
    return segments


def display_masked_img(img, masked_segments):
    masked_img = img.copy()
    masked_img[masked_segments==1] = 0
    masked_img = mark_boundaries(masked_img, masked_segments)
    plt.figure()
    plt.imshow(masked_img)

def fudge(img, segments):
    img = img.copy()
    for segment_id in np.unique(segments):
        img[segments == segment_id] = (
            np.mean(img[segments == segment_id][:, 0]),
            np.mean(img[segments == segment_id][:, 1]),
            np.mean(img[segments == segment_id][:, 2]))
    return img

def sample_around_img(
        img,
        segments,
        scoring_fn,
        n_samples,
        batch_tf=None,
        bsize=10,
        segment_color='fudge',
        has_anchor=None,
        force_full_img=False,
        force_anchor=False,
        state_scores=None):
    if state_scores is None:
        state_scores = {}

    n_segments = np.unique(segments).shape[0]

    # make random masks in the segmentation space
    masks = np.random.randint(0, 2, n_samples * n_segments)
    masks = masks.reshape((n_samples, n_segments))

    if has_anchor is not None:
        masks[:, has_anchor == 1] = 1

    if force_full_img:
        idx_full_img = 0
        masks[idx_full_img, :] = 1 # force first mask to be the full image

    if force_anchor:
        assert has_anchor is not None
        idx_anchor = 1 if force_full_img else 0
        masks[idx_anchor] = has_anchor

    # associate a color to a segment
    # by default, the mean color of the segment
    if segment_color == 'fudge':
        segment_color_img = fudge(img, segments)
    elif segment_color == 'black':
        segment_color_img = img.copy()
        segment_color_img[:] = 0
    else:
        raise ValueError(segment_color)

    targets = [] # used as label in K-Lasso
    features = [] # to calculate similarity with original img
    masked_imgs = []
    for i,row in enumerate(masks):
        row_str = row.tostring()
        if row_str in state_scores:
            saved_score = state_scores[row_str]
            targets.extend(saved_score['target'])
            features.extend(saved_score['feat'])
            continue

        # make masked img using the color of each segment
        zeros = np.where(row == 0)[0]
        mask = np.zeros(segments.shape).astype(bool)
        for z in zeros:
            mask[segments == z] = True
        masked_img = img.copy()
        masked_img[mask] = segment_color_img[mask]
        masked_imgs.append(masked_img)

        last_loop = (i == n_samples-1)
        if last_loop or len(masked_imgs) == bsize:
            if batch_tf is not None:
                masked_imgs = batch_tf(masked_imgs)
            target, feat = scoring_fn(masked_imgs)
            targets.extend(target)
            features.extend(feat)
            state_scores[row_str] = {
                'target': target,
                'feat': feat}
            masked_imgs = []

    targets = np.stack(targets)
    features = np.stack(features)
    return masks, targets, features
