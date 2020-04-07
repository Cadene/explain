import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from .utils import sample_around_img

def score_segments(
        masks,
        targets,
        features,
        distance_fn,
        kernel_fn,
        class_id):
    #target_id = np.argsort(targets[0])[::-1][class_pos]
    target = targets[:,class_id]

    n_features = masks.shape[1]
    distances = distance_fn(masks, targets, features)
    weights = kernel_fn(distances)
    #masks = remove_useless_features(masks, targets, weights)
    masks = preprocessing.scale(masks)
    masks = masks * weights[:,None]
    #lasso = Lasso(alpha=0.01, fit_intercept=False)
    lasso = Ridge(alpha=1., fit_intercept=False)
    lasso.fit(masks, target)#, sample_weight=weights)
    #intercept = lasso.intercept_
    #local_exp = sorted(zip(range(n_features), lasso.coef_), key=lambda x: np.abs(x[1]), reverse=True)
    #training_loss = lasso.score(masks, targets, sample_weight=weights)
    #local_pred = lasso.predict(masks[[0]])
    scores = lasso.coef_
    return scores

def make_masked_segments(
        scores,
        segments,
        n_best_segments):
    best_segment_ids = scores.argsort()[::-1]
    masked_segments = segments.copy()
    masked_segments[:] = 1
    for i in range(n_best_segments):
        eq_ids = segments == best_segment_ids[i]
        masked_segments[eq_ids] = 0
    return masked_segments

def explain_img(
        img,
        scoring_fn,
        segmentation_fn,
        distance_fn,
        kernel_fn,
        class_id,
        n_best_segments=5,
        batch_tf=None,
        bsize=10,
        n_samples=1000,
        segment_color='black'):

    segments = segmentation_fn(img)

    masks, targets, features = sample_around_img(
        img,
        segments,
        scoring_fn,
        n_samples,
        bsize=bsize,
        segment_color=segment_color,
        force_full_img=True)

    scores = score_segments(
        masks,
        targets,
        features,
        distance_fn,
        kernel_fn,
        class_id)

    masked_segments = make_masked_segments(
        scores,
        segments,
        n_best_segments=n_best_segments)

    return masked_segments
