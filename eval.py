import os
import copy
import yaml
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.signal import medfilt
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from resnet import download_resnet, Teacher
from networks import Student_1, Student_2
from dataset import Chunked_sample_dataset


Frame_per_Video = {
    'ped2': {
        0: 0, 1: 180, 2: 360, 3: 510, 4: 690, 5: 840, 6: 1020, 7: 1200, 8: 1380, 9: 1500,
        10: 1650, 11: 1830, 12: 2010
    },
    'avenue': {
        0: 0, 1: 1439, 2: 2650, 3: 3573, 4: 4520, 5: 5527, 6: 6810, 7: 7415, 8: 7451, 9: 8626,
        10: 9467, 11: 9939, 12: 11210, 13: 11759, 14: 12266, 15: 13267, 16: 14007, 17: 14433, 18: 14727, 19: 14975,
        20: 15248, 21: 15324
    },
    'shtech': {
        0: 0, 1: 265, 2: 698, 3: 1035, 4: 1636, 5: 2141, 6: 2550, 7: 3007, 8: 3320, 9: 3729,
        10: 4066, 11: 4403, 12: 4860, 13: 5437, 14: 5750, 15: 6279, 16: 6472, 17: 6761, 18: 7050, 19: 7315,
        20: 7556, 21: 7893, 22: 8182, 23: 8447, 24: 8664, 25: 9097, 26: 9506, 27: 10035, 28: 10348, 29: 10565,
        30: 10806, 31: 11119, 32: 11312, 33: 11577, 34: 11894, 35: 12351, 36: 12688, 37: 13049, 38: 13578, 39: 13987,
        40: 14300, 41: 14685, 42: 15142, 43: 15623, 44: 16080, 45: 16513, 46: 16898, 47: 17139, 48: 17692, 49: 18629,
        50: 19494, 51: 19999, 52: 20312, 53: 20673, 54: 21034, 55: 21563, 56: 21900, 57: 22333, 58: 22814, 59: 23463,
        60: 24112, 61: 24521, 62: 24858, 63: 25627, 64: 26060, 65: 26301, 66: 26518, 67: 26783, 68: 27048, 69: 27265,
        70: 27530, 71: 27939, 72: 28324, 73: 28805, 74: 29262, 75: 29575, 76: 30176, 77: 30417, 78: 30898, 79: 31211,
        80: 31548, 81: 32005, 82: 32222, 83: 32463, 84: 32752, 85: 33089, 86: 33402, 87: 33739, 88: 34004, 89: 34269,
        90: 34606, 91: 34967, 92: 35400, 93: 35641, 94: 36074, 95: 36675, 96: 37180, 97: 37517, 98: 38118, 99: 38383,
        100: 38696, 101: 38937, 102: 39226, 103: 39587, 104: 39972, 105: 40189, 106: 40526, 107: 40791
    }
}


def parse_arguments():
    """
    add argument
    """
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/ped2_recon.yaml')
    parser.add_argument('--model_path', type=str, default='./model/ped2_recon_9932.pt')
    args = parser.parse_args()
    return args


def evaluate(config, model_path):
    """
    evaluate
    """
    download_resnet(config['teacher_name'])
    obj_score = get_object_score(config['test_dataset_dir'], model_path, config['teacher_name'], config['task'], config['k'])
    frame_obj_score = get_frame_obj_score(obj_score, config['test_box_dir'], config['mean'], config['std'], config['k'])
    frame_score = get_frame_score(frame_obj_score, config['dataset_name'], config['num_obj'], config['hybird_para'], config['k'])
    auc = get_frame_auc(frame_score, config['gt_dir'])
    return auc


def get_object_score(test_dataset_dir, student_dir, teacher_name, task, k):
    """
    get score of object
    """
    device = torch.device("cuda:0" if (1 if torch.cuda.is_available() else 0) else "cpu")
    test_files = sorted(os.listdir(test_dataset_dir))
    if k == 1:
        student = Student_1(task).eval().to(device)
    elif k == 2:
        student = Student_2(task).eval().to(device)
    student.load_state_dict(torch.load(student_dir))
    teacher = Teacher(teacher_name).eval().to(device)
    score = [[], [], []]
    for file_idx, file in enumerate(test_files):
        test_dataset = Chunked_sample_dataset(test_dataset_dir + file)
        dataloader_test = DataLoader(test_dataset, batch_size=256, num_workers=0, shuffle=False)
        for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):
            test_batch = test_data[0].to(device)
            with torch.no_grad():
                target = teacher(test_batch[:, 12:])
                if task == 'pred':
                    output = student(test_batch[:, :12])
                elif task == 'recon':
                    output = student(test_batch[:, 12:])
            gen_score = (test_batch[:, 12:] - output[0]) ** 2
            gen_score = torch.mean(torch.mean(torch.mean(gen_score, axis=1), axis=1), axis=1).cpu().data.numpy()
            score[0].append(gen_score)
            for block_num in range(k):
                fea_score = 1 - F.cosine_similarity(target[block_num+1], output[block_num+1])
                fea_score = torch.mean(torch.mean(fea_score, axis=1), axis=1).cpu().data.numpy()
                score[block_num+1].append(fea_score)
    for score_num in range(k+1):
        score[score_num] = np.concatenate(score[score_num], axis=0)
    del dataloader_test
    return score


def get_frame_obj_score(obj_score, test_box_dir, mean, std, k):
    """
    assign object score in same frame
    """
    box = np.load(test_box_dir, allow_pickle=True)
    frame_obj_score = [{}, {}, {}]
    for score_num in range(k+1):
        for ind in range(box.shape[0]):
            frame_obj_score[score_num][ind] = []
        indx = 0
        for i in range(box.shape[0]):
            for j in range(box[i].shape[0]):
                frame_obj_score[score_num][i].append((obj_score[score_num][indx + j]-mean[score_num])/std[score_num])
            indx += box[i].shape[0]
    return frame_obj_score


def zero_object(frame_score):
    """
    set score of frame with no object detected to -1
    only shanghaitech contain frame with no object detected
    """
    for i in range(2):
        for j in range(len(frame_score[i])):
            if len(frame_score[i][j]) == 0:
                frame_score[i][j].append(-1)
    return frame_score


def get_frame_score(frame_obj_score, dataset_name, num_obj, hybird_para, k):
    """
    get frame score
    """
    if dataset_name == 'shtech':
        zero_object(frame_obj_score)
    hybird_frame_obj_score = copy.deepcopy(frame_obj_score[0])
    raw_frame_score = np.array([])
    if k == 1:
        for i in range(len(hybird_frame_obj_score)):
            for j in range(len(hybird_frame_obj_score[i])):
                hybird_frame_obj_score[i][j] = hybird_para[0] * frame_obj_score[0][i][j] \
                                               + hybird_para[1] * frame_obj_score[1][i][j]
    elif k == 2:
        for i in range(len(hybird_frame_obj_score)):
            for j in range(len(hybird_frame_obj_score[i])):
                hybird_frame_obj_score[i][j] = hybird_para[0] * frame_obj_score[0][i][j] \
                                               + hybird_para[1] * frame_obj_score[1][i][j] \
                                               + hybird_para[2] * frame_obj_score[2][i][j]
    for i in hybird_frame_obj_score:
        hybird_frame_obj_score[i].sort()
        raw_frame_score = np.append(raw_frame_score, np.mean(hybird_frame_obj_score[i][-1 * num_obj:]))
    frame_score = copy.deepcopy(raw_frame_score)
    for i in range(len(Frame_per_Video[dataset_name]) - 1):
        frame_score[Frame_per_Video[dataset_name][i]:Frame_per_Video[dataset_name][i + 1]] = \
            medfilt(raw_frame_score[Frame_per_Video[dataset_name][i]:Frame_per_Video[dataset_name][i + 1]], 19)
    return frame_score


def get_frame_auc(frame_score, gt_dir):
    """
    calculate frame auc
    """
    gt = pickle.load(open(gt_dir, "rb"))
    frame_gt = np.array([])
    for i in gt:
        frame_gt = np.append(frame_gt, np.array(gt[i]))
    auc = roc_auc_score(frame_gt, frame_score)
    return auc


if __name__ == '__main__':
    args = parse_arguments()
    config = yaml.safe_load(open(args.config_path))
    print('AUC:' + str(evaluate(config, args.model_path)))
