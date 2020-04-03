# pylint: skip-file
from __future__ import division
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure
import numpy as np
import time

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

MAX_OBJS = 1000


def object_f_score(gt_image, pred_image):
    """
    Calculates the F1 score on object level as described in
    Gland segmentation paper
    :param gt_image:
    :param pred_image:
    :return:
    """
    gt_img = np.array(gt_image)
    gt_img[gt_img > 0] = 1

    pred_img = np.argmax(pred_image, axis=2)
    pred_img[pred_img > 0] = 1

    gt_conts = measure.find_contours(gt_img, 0)
    pred_conts = measure.find_contours(pred_img, 0)

    if len(pred_conts) > MAX_OBJS:
        return np.nan

    gt_objs = []
    pred_objs = []

    for gt_cont, pred_cont in zip(gt_conts, pred_conts):
        gt_cont_idx = gt_cont.astype(np.int16).T
        pred_cont_idx = pred_cont.astype(np.int16).T

        object_img_gt = np.zeros_like(gt_img, dtype=np.int16)
        object_img_pred = np.zeros_like(pred_img, dtype=np.int16)

        object_img_gt[gt_cont_idx[0], gt_cont_idx[1]] = 1  # set contours
        object_img_pred[pred_cont_idx[0], pred_cont_idx[1]] = 1  # set contours

        object_img_gt = binary_fill_holes(object_img_gt).astype(np.int16)
        object_img_pred = binary_fill_holes(object_img_gt).astype(np.int16)

        object_img_gt[gt_cont_idx[0], gt_cont_idx[1]] = 0  # Remove contours
        object_img_pred[pred_cont_idx[0], pred_cont_idx[1]] = 0  # Remove contours

        object_img_gt = object_img_gt[1:-1, 1:-1]  # remove padding
        object_img_pred = object_img_pred[1:-1, 1:-1]  # remove padding

        gt_objs.append(np.argwhere(object_img_gt > 0))
        pred_objs.append(np.argwhere(object_img_pred > 0))

    tp = 0
    fp = 0

    gt_objs = [set(map(tuple, gt_obj)) for gt_obj in gt_objs]
    pred_objs = [set(map(tuple, pred_obj)) for pred_obj in pred_objs]

    for pred_obj in pred_objs:
        if len(pred_obj) == 0:  # caution of empty objects
            continue
        match = 0
        for gt_obj in gt_objs:
            if len(gt_obj) == 0:
                continue
            if (float(len(pred_obj)) / len(gt_obj)) > 0.5:  # This is for speeding up comparisons
                intersection = list(pred_obj & gt_obj)

                if (len(intersection) / len(gt_obj)) > 0.5:  # Intersects with more than 50 % then tp += 1 :
                    match = 1
                    break

        if match == 1:
            tp += 1
        else:
            fp += 1
    fn = len(gt_objs) - tp

    precision = tp / float(tp + fp) if tp > 0 else 0
    recall = tp / float(tp + fn) if tp > 0 else 0

    if (precision + recall) == 0:
        print("gt_objs: " + str(len(gt_objs)) + " pred_objs: " + str(len(pred_objs)))
        print("precision: " + str(precision) + " recall: " + str(recall))
        print("fn: " + str(fn) + " tp: " + str(tp) + " fp: " + str(fp))
        return 0

    return 2 * precision * recall / (precision + recall)


def create_objects_from_seg(img):
    result = np.empty(shape=(img.shape[0], img.shape[1], 0))
    padded_img = np.pad(img, 1, 'constant')
    objects = measure.find_contours(padded_img, 0)

    if len(objects) > MAX_OBJS:
        return np.nan

    for i in objects:
        obj_idx = np.array(i, dtype=np.int16).T
        object_img = np.zeros_like(padded_img, dtype=np.int16)
        object_img[obj_idx[0], obj_idx[1]] = 1  # set contours
        object_img = binary_fill_holes(object_img).astype(np.int16)
        object_img[obj_idx[0], obj_idx[1]] = 0  # Remove contours
        object_img = object_img[1:-1, 1:-1]  # remove padding
        object_img = np.expand_dims(object_img, axis=2)
        result = np.concatenate((result, object_img), axis=2)

    return result


def preds_to_sets(preds):
    object_sets = {}
    for i in range(preds.shape[-1]):
        obj = preds[:, :, i]
        idcs = np.argwhere(obj > 0)
        if len(idcs) > 0:
            idcs_set = set(map(tuple, idcs))
            object_sets[i] = idcs_set

    return object_sets


def ground_truth_to_sets(img):
    object_sets = {}

    for i, value in enumerate(np.unique(img)[1:]):
        idcs = np.argwhere(img == value)
        idcs_set = set(map(tuple, idcs))
        object_sets[i] = idcs_set

    return object_sets


def find_most_overlapping_object(dict_of_objects, object):
    highest_itersection = 0
    highest_itersection_idx = None
    for i, obj in dict_of_objects.iteritems():
        itc = len(object.intersection(obj))
        if itc >= highest_itersection:
            highest_itersection = itc
            highest_itersection_idx = i

    return highest_itersection_idx


def dice(G, S):
    # 2* |intersection of G and S| / |G| + |S|
    score = (2 * len(G.intersection(S))) / (len(G) + len(S))
    return score


def dice_object_score(ground_truth_img, prediction):
    pred = np.argmax(prediction, axis=2)
    pred = create_objects_from_seg(pred)
    if np.isnan(pred).any():  # If we have too many objects return nan
        return pred

    pred_object_sets = preds_to_sets(pred)
    gt_object_sets = ground_truth_to_sets(ground_truth_img)

    ng_sum = 0
    for i, obj in gt_object_sets.iteritems():
        gs = gamma_sigma(obj, gt_object_sets.values())
        best_pred_obj_idx = find_most_overlapping_object(pred_object_sets, obj)
        dice_score = dice(pred_object_sets[best_pred_obj_idx], obj)
        ng_sum += gs * dice_score

    ns_sum = 0
    for i, obj in pred_object_sets.iteritems():
        gs = gamma_sigma(obj, pred_object_sets.values())
        best_gt_obj_idx = find_most_overlapping_object(gt_object_sets, obj)
        dice_score = dice(gt_object_sets[best_gt_obj_idx], obj)
        ns_sum += gs * dice_score

    return 0.5 * (ng_sum + ns_sum)


def gamma_sigma(pixel_set, list_of_pixel_sets):
    return len(pixel_set) / sum(len(gj) for gj in list_of_pixel_sets)


def hausdorff_score(G, S):
    G_list = [np.array(g) for g in list(G)]
    S_list = [np.array(s) for s in list(S)]

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(G_list)
    dist1, idx = nn.kneighbors(S_list)
    nn2 = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn2.fit(S_list)
    dist2, idx = nn2.kneighbors(G_list)

    dist1 = max(dist1)
    dist2 = max(dist2)

    return max(dist1, dist2)


def hausdorff_object_score(ground_truth_img, prediction):
    pred = np.argmax(prediction, axis=2)
    pred = create_objects_from_seg(pred)
    if np.isnan(pred).any():  # if we find too many objects
        return pred

    pred_object_sets = preds_to_sets(pred)
    gt_object_sets = ground_truth_to_sets(ground_truth_img)
    ng_sum = 0
    for i, obj in gt_object_sets.iteritems():
        gs = gamma_sigma(obj, gt_object_sets.values())
        best_pred_obj_idx = find_most_overlapping_object(pred_object_sets, obj)
        h_score = hausdorff_score(pred_object_sets[best_pred_obj_idx], obj)
        ng_sum += gs * h_score

    ns_sum = 0
    for i, obj in pred_object_sets.iteritems():
        gs = gamma_sigma(obj, pred_object_sets.values())
        best_gt_obj_idx = find_most_overlapping_object(gt_object_sets, obj)
        h_score = hausdorff_score(gt_object_sets[best_gt_obj_idx], obj)
        ns_sum += gs * h_score

    return 0.5 * (ng_sum + ns_sum)


def get_scores(predictions, ground_truth, dataset_name="test", loop_number=0, hps=None):
    hausdorff_scores = []
    f1_scores = []
    dice_scores = []

    for i, gt_img in enumerate(ground_truth):

        if i == 0:  # To save as example out
            gt_example = ground_truth[i, :, :]
            example_img = predictions[i, :, :, :]

        pred_img = predictions[i, :, :, :]

        hausdorff_scores.append(hausdorff_object_score(gt_img, pred_img))
        f1_scores.append(object_f_score(gt_img, pred_img))
        dice_scores.append(dice_object_score(gt_img, pred_img))

    metrics_list = [np.array(hausdorff_scores), np.array(f1_scores), np.array(dice_scores)]

    metrics = tuple(0 if np.isnan(metric).any() else np.mean(metric) for metric in metrics_list)

    _print_evaluation(dataset_name, metrics, image_prediction=(gt_example, example_img), loop_number=loop_number, hps=hps)
    #_print_evaluation_debugging(dataset_name, (hausdorff_scores, f1_scores, dice_scores))


def _print_evaluation(dataset_name, metrics, image_prediction=None, loop_number=0, hps=None):
    img = (np.argmax(image_prediction[1], axis=2) * 255).astype(np.uint8)
    img = Image.fromarray(img, mode='L')
    img.save('./' + hps.get('exp_name') + "/" + dataset_name + '_' + str(loop_number) + '.bmp')

    gt_img = (image_prediction[0] * 255).astype(np.uint8)
    gt_img = Image.fromarray(gt_img, mode='L')
    gt_img.save('./' + hps.get('exp_name') + "/" + dataset_name + '_' + str(loop_number) + '_gt.bmp')

    hausdorf = metrics[0]
    f1 = metrics[1]
    dice = metrics[2]

    print('\nF_score: {}, Hausdorff: {}, Dice: {}\n'.format(f1, hausdorf, dice))
    t = int(round(time.time() * 1000))

    with open('./' + hps.get('exp_name') + "/" + dataset_name + '.csv', 'a') as file:
        file.write('{}, {}, {}, {}\n'.format(t, f1, hausdorf, dice))


def _print_evaluation_debugging(dataset_name, metrics):
    hausdorf = metrics[0]
    f1 = metrics[1]
    dice = metrics[2]

    with open('./' + self.hps.get('exp_name') + "/" + dataset_name + '_hausdorf.csv', 'a') as file:
        csv_string = ["{}" for _ in hausdorf]
        csv_string = ", ".join(csv_string) + "\n"
        file.write(csv_string.format(*hausdorf))

    with open('./' + self.hps.get('exp_name') + "/" + dataset_name + '_dice.csv', 'a') as file:
        csv_string = ["{}" for _ in dice]
        csv_string = ", ".join(csv_string) + "\n"
        file.write(csv_string.format(*dice))

    with open('./' + self.hps.get('exp_name') + "/" + dataset_name + '_f1.csv', 'a') as file:
        csv_string = ["{}" for _ in f1]
        csv_string = ", ".join(csv_string) + "\n"
        file.write(csv_string.format(*f1))
