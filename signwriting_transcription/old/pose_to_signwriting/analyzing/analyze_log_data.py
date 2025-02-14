import argparse
import re
from pathlib import Path

import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm

COLORS = ['b', 'r', 'g', 'y', 'm', 'c', 'k']  # colors for the plots


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True, type=str)
    parser.add_argument("--trg-dir", required=True, type=str)
    args_val = parser.parse_args()
    return args_val


def extract_training_dev(file):
    lines = file.readlines()

    log_info = {'steps': [], 'training_losses': [], 'validation_losses': []}
    batch_losses = []
    current_step = 0

    for line in lines:
        # extract training loss
        match_train = re.search(r'Epoch\s+\d+,\s+Step:\s+(\S+)\d+,\s+Batch Loss:\s+(\S+),', line)
        if match_train:
            batch_loss = float(match_train.group(2))
            batch_losses.append(batch_loss)
            current_step = float(match_train.group(1))
        # extract validation loss
        match_val = re.search(r'Evaluation result \(greedy\) fsw_eval:\s+\S+,\s+loss:\s+(\S+),', line)
        if match_val:
            validation_loss = float(match_val.group(1))
            log_info['validation_losses'].append(validation_loss)
            log_info['training_losses'].append(
                sum(batch_losses) / len(batch_losses))    # average batch loss over interval
            log_info['steps'].append(current_step)
            batch_losses = []   # reset the batch losses for the next interval

    return log_info['steps'], log_info['training_losses'], log_info['validation_losses']


def extract_fsw_score(file):
    lines = file.readlines()

    current_step = 0
    steps = []
    fsw_scores = []

    for line in lines:
        # track over the steps
        match_train = re.search(r'Epoch\s+\d+,\s+Step:\s+(\S+)\d+,\s+Batch Loss:\s+\S+,', line)
        if match_train:
            current_step = float(match_train.group(1))
        # extract fsw score
        match_val = re.search(r'Evaluation result \(greedy\) fsw_eval:\s+(\S+),\s+loss:\s+\S+,', line)
        if match_val:
            fsw_score = float(match_val.group(1))
            fsw_scores.append(fsw_score)
            steps.append(current_step)

    return steps, fsw_scores


def plot_losses(steps, steps_info, title):
    fig, axis = plt.subplots()
    # iterate over the measures and plot them
    for index, (measure, measure_info) in enumerate(steps_info.items()):
        axis.plot(steps, measure_info, label=measure, marker='o', color=COLORS[index % len(COLORS)])

    axis.set_xlabel('Steps')
    axis.set_ylabel('Loss')
    axis.set_title(title)
    axis.legend(loc='center right')
    return fig


def main(log_dir, trg_dir):
    print('Analyzing log files ...')
    log_dir = Path(log_dir)
    trg_dir = Path(trg_dir)

    for log_file in tqdm(log_dir.glob("*.log")):
        with open(log_file, 'r', encoding='utf-8') as file:
            steps, training_losses, validation_loss = extract_training_dev(file)
            file.seek(0)    # reset the file pointer
            _, fsw_score = extract_fsw_score(file)

            # create the plots
            train_dev_info = {'training losses': training_losses, 'validation loss': validation_loss}
            fsw_score_info = {'fsw score': fsw_score}
            dev_training_plot = plot_losses(steps, train_dev_info, 'training and validation losses over intervals')
            fsw_score_plot = plot_losses(steps, fsw_score_info, 'fsw score over intervals')

            # save the plots
            trg_file = trg_dir / log_file.stem
            dev_training_plot.savefig(f'{trg_file} - training_validation_losses.png')
            fsw_score_plot.savefig(f'{trg_file} - fsw_score.png')


if __name__ == '__main__':
    args = get_args()
    main(args.logs_dir, args.trg_dir)
