import numpy as np
from .utils import sample_around_img

def compute_beta(n_features, n_iter, delta):
    alpha = 1.1
    k = 405.5
    temp = np.log(k * n_features * (n_iter ** alpha) / delta)
    beta = temp + np.log(temp)
    return beta

def kl_bernoulli(p, q):
    p = min(0.9999999999999999, max(0.0000001, p))
    q = min(0.9999999999999999, max(0.0000001, q))
    return (p * np.log(float(p) / q) + (1 - p) *
            np.log(float(1 - p) / (1 - q)))

def dlow_bernoulli(p, level):
    um = p
    lm = max(min(1, p - np.sqrt(level / 2.)), 0)
    for j in range(1, 17):
        qm = (um + lm) / 2.
#         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
    return lm

def dup_bernoulli(p, level):
    lm = p
    um = min(min(1, p + np.sqrt(level / 2.)), 1)
    for j in range(1, 17):
        qm = (um + lm) / 2.
#         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
    return um

def coverage(anchor):
    # the smallest anchor is the one which covers the most input space
    n_segments = anchor.shape[0]
    anchor_size = anchor[anchor == 1].shape[0]
    cov = (n_segments-anchor_size+1)/n_segments
    # cov \in \[0.1, 1\]
    return cov

def generate_candidates(
        anchors,
        n_segments):
    candidates = []

    if len(anchors) == 0:
        for i in range(n_segments):
            a = np.zeros(n_segments)
            a[i] = 1
            candidates.append(a)
        return candidates

    anchors_set = {}
    for a in anchors:
        for i in range(n_segments):
            if a[i] == 0:
                candidate = a.copy()
                candidate[i] = 1
                c_str = candidate.tostring()
                if c_str not in anchors_set:
                    anchors_set[c_str] = True
                    candidates.append(candidate)
    return candidates

def find_anchors(
        candidates,
        img,
        class_id,
        segments,
        scoring_fn,
        n_samples=20,
        bsize=10,
        segment_color='fudge',
        B=10,
        tau=0.95,
        eps=0.1,
        delta=0.05):

    def sample_fn(n_samples, anchor, state, force_anchor=False):
        _, targets, _ = sample_around_img(
            img,
            segments,
            scoring_fn,
            n_samples,
            bsize=bsize,
            segment_color=segment_color,
            has_anchor=anchor,
            force_anchor=force_anchor,
            state_scores=state['scores'])
        argmax_ = targets.argmax(1)
        n_success = len(argmax_[argmax_ == class_id])
        return n_success

    n_segments = len(np.unique(segments))
    n_candidates = len(candidates)
    n_iter = 1
    u_bounds = np.zeros(n_candidates)
    l_bounds = np.zeros(n_candidates)

    #if state is None:
    state = {
        'n_samples': np.zeros(n_candidates),
        'n_success': np.zeros(n_candidates),
        'scores': {}
    }
    for aid, a in enumerate(candidates):
        state['n_samples'][aid] += 1
        state['n_success'][aid] += sample_fn(1, a, state, True)

    def update_candidates_bounds(
            u_bounds, l_bounds,
            n_iter, state,
            delta, B, n_segments):
        beta = compute_beta(n_segments, n_iter, delta)
        precs = state['n_success'] / state['n_samples']
        argsort_ = np.argsort(precs)
        l_cids = argsort_[-B:] # B of best candidates
        u_cids = argsort_[:-B] # rest of candidates
        for idx in u_cids:
            u_bounds[idx] = dup_bernoulli(
                precs[idx],
                beta / state['n_samples'][idx])
        for idx in l_cids:
            l_bounds[idx] = dlow_bernoulli(
                precs[idx],
                beta / state['n_samples'][idx])
        u_cid = u_cids[np.argmax(u_bounds[u_cids])]
        l_cid = l_cids[np.argmin(l_bounds[l_cids])]
        # upper bounds, lower bounds
        # candidate id of max upper bound and min lower bound
        return u_bounds, l_bounds, u_cid, l_cid

    u_bounds, l_bounds, u_cid, l_cid = update_candidates_bounds(
        u_bounds, l_bounds,
        n_iter, state,
        delta, B, n_segments)

    prec_u_b_a_p = u_bounds[u_cid]
    prec_l_b_a = l_bounds[l_cid]

    a_p = candidates[u_cid]
    a = candidates[l_cid]

    while prec_u_b_a_p - prec_l_b_a > eps:

        state['n_samples'][l_cid] += n_samples
        state['n_success'][l_cid] += sample_fn(n_samples, a, state, True)

        state['n_samples'][u_cid] += n_samples
        state['n_success'][u_cid] += sample_fn(n_samples, a_p, state, True)

        # z ~ D(z|a)
        # z_p ~ D(z_p|a_p)
        u_bounds, l_bounds, u_cid, l_cid = update_candidates_bounds(
            u_bounds, l_bounds,
            n_iter, state,
            delta, B, n_segments)

        # update precisions
        prec_u_b_a_p = u_bounds[u_cid]
        prec_l_b_a = l_bounds[l_cid]

        # update candidates
        a_p = candidates[u_cid]
        a = candidates[l_cid]
        
        verbose = True
        if verbose:
            prec_u_b_a_p_str = str(prec_u_b_a_p)[:5]
            prec_l_b_a_str = str(prec_l_b_a)[:5]
            print(f'n_iter:{n_iter}  upper_bound_second_best:{prec_u_b_a_p_str} lower_bound_best:{prec_l_b_a_str}')
        # print('prec_u_b_a_p', prec_u_b_a_p)
        # print('prec_l_b_a', prec_l_b_a)
        # print('prec_u_b_a_p - prec_l_b_a > eps', prec_u_b_a_p - prec_l_b_a > eps)
        # print('prec_l_b_a <= tau', prec_l_b_a <= tau)
        # print('prec_u_b_a_p >= tau', prec_u_b_a_p >= tau)
        # print()

        n_iter += 1

    precs = state['n_success'] / state['n_samples']
    best_cids = np.argsort(precs)[-B:]

    best_candidates = [candidates[cid] for cid in best_cids]
    best_precs = [precs[cid] for cid in best_cids]

    beta = np.log(1 / (delta / (1 + (B - 1) * n_segments)))

    valid_cids = []
    for cid, a, p in zip(best_cids, best_candidates, best_precs):
        while (p >= tau and l_bounds[cid] < tau) or \
              (p < tau and u_bounds[cid] >= tau):

            state['n_samples'][cid] += n_samples
            state['n_success'][cid] += sample_fn(n_samples, a, state, True)

            precs = state['n_success'] / state['n_samples']
            u_bounds[cid] = dup_bernoulli(
                precs[cid],
                beta / state['n_samples'][cid])
            l_bounds[cid] = dlow_bernoulli(
                precs[cid],
                beta / state['n_samples'][cid])

            verbose = True
            if verbose:
                print(cid, u_bounds[cid], l_bounds[cid])

        #if l_bounds[cid] >= tau:
        valid_cids.append(cid)

    valid_candidates = [candidates[cid] for cid in valid_cids]
    valid_precs = [precs[cid] for cid in valid_cids]
    return valid_candidates, valid_precs


# def find_anchor_star(anchors):
#     covs = np.array([coverage(a) for a in anchors])
#     argmax_ = covs.argmax()
#     a_star = anchors[argmax_]
#     return a_star


def score_segments(
        img,
        segments,
        scoring_fn,
        class_id,
        n_samples=20,
        bsize=10,
        segment_color='fudge',
        B=10,
        tau=0.95,
        eps=0.1,
        delta=0.05):

    n_segments = len(np.unique(segments))
    anchors = []

    anchor_size = 1
    while anchor_size < n_segments:
        candidates = generate_candidates(
            anchors,
            n_segments)

        assert(len(candidates) != 0)

        anchors, precisions = find_anchors(
            candidates,
            img,
            class_id,
            segments,
            scoring_fn,
            n_samples=n_samples,
            bsize=bsize,
            segment_color=segment_color,
            B=B,
            tau=tau,
            eps=eps,
            delta=delta)

        # no better candidate found
        if len(anchors) == 0:
            print('No valid candidates found during best arm identification. Stopping search.')
            break

        verbose = False
        if verbose:
            from .utils import display_masked_img
            for a, p in zip(anchors, precisions):
                print('precision', p)
                masked_segments = make_masked_segments(
                    a,
                    segments)
                display_masked_img(img, masked_segments)

        best_precision = max(precisions)
        if best_precision >= tau:
            argmax_ = np.array(precisions).argmax()
            a_star = anchors[argmax_]
            break
        #a_star = find_anchor_star(anchors)

        # # equivalent to if anchor_size == 1
        #     if coverage(a_star) == 1:
        #         break

        anchor_size += 1

    return a_star

def make_masked_segments(
        scores,
        segments):
    n_segments = scores.shape[0]
    segment_ids = np.array(range(n_segments))
    best_segment_ids = segment_ids[scores == 1]
    masked_segments = segments.copy()
    masked_segments[:] = 1
    for idx in best_segment_ids:
        masked_segments[segments == idx] = 0
    return masked_segments

def explain_img(
        img,
        scoring_fn,
        segmentation_fn,
        #distance_fn,
        #kernel_fn,
        class_id,
        bsize=10,
        segment_color='black',
        B=10,
        tau=0.98,
        eps=0.1,
        delta=0.05,
        n_samples=100):

    segments = segmentation_fn(img)

    scores = score_segments(
        img,
        segments,
        scoring_fn,
        class_id,
        B=B,
        tau=tau,
        eps=eps,
        delta=delta,
        n_samples=n_samples)

    masked_segments = make_masked_segments(
        scores,
        segments)

    return masked_segments
