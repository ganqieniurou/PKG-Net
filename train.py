import os
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from eval import evaluate
from resnet import download_resnet, Teacher
from dataset import Chunked_sample_dataset
from loss import Gradient_Loss, Generation_Loss, Feature_Loss
from networks import Student_1, Student_2, weights_init_kaiming


def parse_arguments():
    """
    add argument
    """
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/avenue.yaml')
    args = parser.parse_args()
    return args


def train(args):
    config = yaml.safe_load(open(args.config_path))
    download_resnet(config['teacher_name'])
    student_train(config)


def student_train(config):
    """
    student network training
    """
    (train_dataset_dir, teacher_name, dataset_name,
     learning_rate, milestone, lr_decay, save_dir,
     batch_size, task, max_epoch, loss_para, k) = \
        (config['train_dataset_dir'], config['teacher_name'], config['dataset_name'],
         config['learning_rate'], config['milestone'], config['lr_decay'], config['save_dir'],
         config['batch_size'], config['task'], config['max_epoch'], config['loss_para'], config['k'])

    device = torch.device("cuda:0" if (1 if torch.cuda.is_available() else 0) else "cpu")
    print(f'Device used: {device}')
    train_files = sorted(os.listdir(train_dataset_dir))
    if k == 1:
        student = Student_1(task).eval().to(device)
    elif k == 2:
        student = Student_2(task).eval().to(device)

    if os.path.exists(save_dir):
        student.load_state_dict(torch.load(save_dir))
        print('Loading model from ' + save_dir)
    else:
        student.apply(weights_init_kaiming)
        print('weights initializing')
    teacher = Teacher(teacher_name).eval().to(device)

    generation_loss = Generation_Loss().to(device)
    gradient_loss = Gradient_Loss(3, 3, device).to(device)
    feature_loss = Feature_Loss()

    optimizer = optim.Adam(student.parameters(), lr=learning_rate, eps=1e-7, weight_decay=0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone], gamma=lr_decay)

    max_auc = -1
    best_model = 0

    for epoch in range(max_epoch):
        running_loss = 0
        for file_idx, file in enumerate(train_files):
            train_dataset = Chunked_sample_dataset(train_dataset_dir + file)
            dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                                    shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, file_idx),
                                        total=len(dataloader)):
                optimizer.zero_grad()

                train_batch = train_data[0].to(device)
                with torch.no_grad():
                    target = teacher(train_batch[:, 12:])
                if task == 'pred':
                    output = student(train_batch[:, :12])
                elif task == 'recon':
                    output = student(train_batch[:, 12:])

                gen_loss = generation_loss(train_batch[:, 12:], output[0])
                grad_loss = gradient_loss(train_batch[:, 12:], output[0])
                fea_loss = feature_loss(target, output, k)
                loss = loss_para['gen'] * gen_loss + \
                       loss_para['fea'] * fea_loss + \
                       loss_para['grad'] * grad_loss
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

        print(f"Epoch {epoch + 1} \t loss: {running_loss}")
        scheduler.step()

        torch.save(student.state_dict(), save_dir)
        print(f"Student model saved.")

        if epoch >= 0:
            auc = evaluate(config, save_dir)
            auc = round(auc, 4)
            if auc > max_auc:
                max_auc = auc
                if best_model != 0:
                    os.remove(old_best_model)
                best_model = f"./model/{dataset_name}_{task}_{str(auc)[2:6]}.pt"
                torch.save(student.state_dict(), best_model)
                old_best_model = best_model
            print(max_auc)

if __name__ == '__main__':
    args = parse_arguments()
    train(args)
