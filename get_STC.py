import numpy as np
import os
from argparse import ArgumentParser
import cv2
import torch
from tqdm import tqdm
from dataset import get_dataset, img_tensor2numpy, img_batch_tensor2numpy
from mmdet.apis import init_detector, inference_detector
import joblib

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DATASET_CFGS = {
    "ped2": {"conf_thr": 0.5, "min_area": 10 * 10, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 3,
             'contour_min_area': 10 * 10},
    "avenue": {"conf_thr": 0.25, "min_area": 40 * 40, "cover_thr": 0.6, "binary_thr": 18, "gauss_mask_size": 5,
               'contour_min_area': 40 * 40},
    "shanghaitech": {"conf_thr": 0.5, "min_area": 8 * 8, "cover_thr": 0.65, "binary_thr": 15, "gauss_mask_size": 5,
                     'contour_min_area': 40 * 40}
}


def parse_arguments():
    """
    add argument
    """
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='shtech')
    parser.add_argument('--dataset_path', type=str, default='./preprocess/shtech')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--ROI_path', type=str, default='notgiven')
    args = parser.parse_args()
    return args


def get_ROI(args):
    if args.ROI_path == 'notgiven':
        ROI_path = getROIregion(args.dataset_name, args.dataset_path, args.mode)
        getobject(args.dataset_name, args.dataset_path, args.mode, ROI_path)
    else:
        getobject(args.dataset_name, args.dataset_path, args.mode, args.ROI_path)


def getObjBboxes(img, model, dataset_name):
    result = inference_detector(model, img)
    CONF_THR = DATASET_CFGS[dataset_name]["conf_thr"]
    MIN_AREA = DATASET_CFGS[dataset_name]["min_area"]

    bbox_result = result
    bboxes = np.vstack(bbox_result)

    scores = bboxes[:, -1]  # x1,y1,x2,y2,class_score
    bboxes = bboxes[scores > CONF_THR, :]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    return bboxes[bbox_areas >= MIN_AREA, :4]


def delCoverBboxes(bboxes, dataset_name):
    assert bboxes.ndim == 2
    assert bboxes.shape[1] == 4

    COVER_THR = DATASET_CFGS[dataset_name]["cover_thr"]

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    bbox_areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    sort_idx = bbox_areas.argsort()  # Index of bboxes sorted in ascending order by area size

    keep_idx = []
    for i in range(sort_idx.size):  # calculate overlap with i-th bbox
        # Calculate the point coordinates of the intersection
        x11 = np.maximum(x1[sort_idx[i]], x1[sort_idx[i + 1:]])
        y11 = np.maximum(y1[sort_idx[i]], y1[sort_idx[i + 1:]])
        x22 = np.minimum(x2[sort_idx[i]], x2[sort_idx[i + 1:]])
        y22 = np.minimum(y2[sort_idx[i]], y2[sort_idx[i + 1:]])
        # Calculate the intersection area
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h

        ratios = overlaps / bbox_areas[sort_idx[i]]
        num = ratios[ratios > COVER_THR]
        if num.size == 0:
            keep_idx.append(sort_idx[i])

    return bboxes[keep_idx]


def getFgBboxes(img_batch, bboxes, dataset_name):
    area_thr = DATASET_CFGS[dataset_name]["contour_min_area"]
    binary_thr = DATASET_CFGS[dataset_name]["binary_thr"]
    gauss_mask_size = DATASET_CFGS[dataset_name]["gauss_mask_size"]
    extend = 2

    sum_grad = 0
    for i in range(img_batch.shape[0] - 1):
        img1 = img_batch[i, :, :, :]
        img2 = img_batch[i + 1, :, :, :]
        img1 = cv2.GaussianBlur(img1, (gauss_mask_size, gauss_mask_size), 0)
        img2 = cv2.GaussianBlur(img2, (gauss_mask_size, gauss_mask_size), 0)

        grad = cv2.absdiff(img1, img2)
        sum_grad = grad + sum_grad

    sum_grad = cv2.threshold(sum_grad, binary_thr, 255, cv2.THRESH_BINARY)[1]  # temporal gradient

    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
        extend_y1 = np.maximum(0, bbox_int[1] - extend)
        extend_y2 = np.minimum(bbox_int[3] + extend, sum_grad.shape[0])
        extend_x1 = np.maximum(0, bbox_int[0] - extend)
        extend_x2 = np.minimum(bbox_int[2] + extend, sum_grad.shape[1])
        sum_grad[extend_y1:extend_y2 + 1, extend_x1:extend_x2 + 1] = 0

    sum_grad = cv2.cvtColor(sum_grad, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(sum_grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg_bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        sum_grad = cv2.rectangle(sum_grad, (x, y), (x + w, y + h), color=255, thickness=1)
        area = (w + 1) * (h + 1)
        if area > area_thr and w / h < 10 and h / w < 10:
            extend_x1 = np.maximum(0, x - extend)
            extend_y1 = np.maximum(0, y - extend)
            extend_x2 = np.minimum(x + w + extend, sum_grad.shape[1])
            extend_y2 = np.minimum(y + h + extend, sum_grad.shape[0])
            fg_bboxes.append([extend_x1, extend_y1, extend_x2, extend_y2])

    return np.array(fg_bboxes)


def getROIregion(dataset_name, dataset_path, mode):
    mm_det_config_file = 'preprocess/assets/cascade_rcnn_r101_fpn_1x_coco.py'
    mm_det_ckpt_file = 'preprocess/assets/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth'

    dataset = get_dataset(dataset_name=dataset_name,
                          dir=dataset_path,
                          context_frame_num=1, mode=mode)

    mm_det_model = init_detector(mm_det_config_file, mm_det_ckpt_file, device="cuda:0")

    all_bboxes = list()
    print('extracting STC')
    for idx in tqdm(range(len(dataset)), total=len(dataset)):
        batch, _ = dataset.__getitem__(idx)

        # centric frame
        cur_img = img_tensor2numpy(batch[1])
        obj_bboxes = getObjBboxes(cur_img, mm_det_model, dataset_name)
        # filter some overlapped bbox
        obj_bboxes = delCoverBboxes(obj_bboxes, dataset_name)
        fg_bboxes = getFgBboxes(img_batch_tensor2numpy(batch), obj_bboxes, dataset_name)
        if fg_bboxes.shape[0] > 0:
            cur_bboxes = np.concatenate((obj_bboxes, fg_bboxes), axis=0)
        else:
            cur_bboxes = obj_bboxes
        all_bboxes.append(cur_bboxes)
    save_path = f'./data/{dataset_name}/{dataset_name}_{mode}.npy'
    np.save(save_path, all_bboxes)
    print('STC saved')
    return save_path


def getobject(dataset_name, dataset_path, mode, save_path):
    global_sample_id = 0
    cnt = 0
    chunk_id = 0
    save_dir = f'./data/{dataset_name}/{mode}/'
    chunked_samples = dict(sample_id=[], appearance=[])
    ROI = np.load(save_path, allow_pickle=True)

    dataset = get_dataset(
        dataset_name=dataset_name,
        dir=dataset_path,
        context_frame_num=4, mode=mode,
        border_mode="predict", all_bboxes=ROI,
        patch_size=32, of_dataset=False
    )

    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print('Extracting foreground in {}-th frame, {} in total'.format(idx + 1, len(dataset)))

        batch, _ = dataset.__getitem__(idx)
        cur_bboxes = ROI[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            for idx_box in range(cur_bboxes.shape[0]):
                chunked_samples["sample_id"].append(global_sample_id)
                chunked_samples["appearance"].append(batch[idx_box])
                global_sample_id += 1
                cnt += 1

                if cnt == 200000:
                    chunked_samples["sample_id"] = np.array(chunked_samples["sample_id"])
                    chunked_samples["appearance"] = np.array(chunked_samples["appearance"])
                    joblib.dump(chunked_samples, os.path.join(save_dir, f'{dataset_name}_{mode}_{chunk_id}.pkl'))
                    print("Chunk %d file saved!" % chunk_id)

                    chunk_id += 1
                    cnt = 0
                    del chunked_samples
                    chunked_samples = dict(sample_id=[], appearance=[])

    if len(chunked_samples["sample_id"]) != 0:
        chunked_samples["sample_id"] = np.array(chunked_samples["sample_id"])
        chunked_samples["appearance"] = np.array(chunked_samples["appearance"])
        joblib.dump(chunked_samples, os.path.join(save_dir, f'{dataset_name}_{mode}_{chunk_id}.pkl'))
        print("Chunk %d file saved!" % chunk_id)

    print('All samples have been saved!')


if __name__ == '__main__':
    args = parse_arguments()
    get_ROI(args)
