from machine_learning.modules import LSTMGenerator, generate_movement_sequence, load_data, device
from machine_learning.human_data_ingestion import MAX_MOVEMENT_SEQUENCE_LENGTH
import torch
from balabit_feature_extraction.rawdata2actions import processSession1
import matplotlib.pyplot as plt
import numpy as np
import math
import random
def generate_samples(paths, data, num_samples=1):
    lstms = {path: LSTMGenerator() for path in paths}
    examples = {}
    for path in lstms:
        lstm = lstms[path].to(device)
        lstm.load_state_dict(torch.load(path))
        lstm.eval()
        print(f"Path {path}")
        examples[path] = [generate_movement_sequence(lstm, MAX_MOVEMENT_SEQUENCE_LENGTH, data) for _ in range(num_samples)]

    return examples
def remove_adjacent(nums):
  i = 1
  while i < len(nums):
    if nums[i] == nums[i-1]:
      nums.pop(i)
      i -= 1
    i += 1
  return nums

def convert_to_action_sequence(movement_sequence):
    movmts = []
    timestamp = 0
    prev_xy = (0, 0)
    # due to a preprocessing error we must adjust these
    min_x = np.abs(min([x[0].item() for x in movement_sequence.squeeze()]))
    for movement in movement_sequence:
        x, y = tuple(movement.squeeze()[:2])
        x += min_x

        # we scaled down the coordinates to make our losses more reasonably sized
        x = int(x.item() * 3999)
        y = int(y.item() * 3999)
        x = min(max(x, 1), 3999)
        y = min(max(y, 1), 3999)
        # the architecture doesn't add time stamps so we will need to add them; this is the average time step
        timestamp += 0.29

        # additionally the generator resulted in very bad button action sequences so we will ignore those and replace all movements with simple moves
        state, button = ('Move', 'NoButton')

        mvmt = {"client timestamp": timestamp, "x": x, "y": y, "state": state, "button": button}
        if prev_xy != (x,y):
            movmts.append(mvmt)

        prev_xy = (x, y)

    movmts[-1]["state"] = "Pressed"
    movmts[-1]["button"] = "Left"

    actions = processSession1(movmts)
    if actions == None or len(actions) == 0:
        return None
    else:
        return actions[0]

feature_keys = ["Action type", "Trajectory length", "Time duration", "Direction",
              "Straightness", "Number of points", "Sum of angles",
              "Mean curvature", "SD curvature", "Max curvature", "Minimum curvature",
              "Mean angle", "SD angle", "Max angle", "Min angle",
              "Largest distance between points",
              "Distance end to end", "Numer of cirtical points",
              "Mean vx", "SD vx", "Max vx", "Min vx",
              "Mean vy", "SD vy", "Max vy", "Min vy",
              "Mean v", "SD v", "Max v", "Min v",
              "Mean a", "SD a", "Max a", "Min a",
              "Mean jerk", "SD jerk", "Max jerk", "Min jerk",
              "Duration of first segment with acceleration"]

image_titles = ["GAN 1", "GAN 2", "Simple pretraining", "Full training; no GAN",  "Human data"]

def print_examples(name, sequences):
    example_seqs = random.sample(sequences, 5)
    for idx, actions in enumerate(example_seqs):
        with open(f"machine_learning/experiment_logs/{name}_{idx}.csv", "w+") as f:
            min_x = np.abs(min([x[0].item() for x in actions.squeeze()]))
            for movement in actions:
                x, y = tuple(movement.squeeze()[:2])
                x += min_x

                # we scaled down the coordinates to make our losses more reasonably sized
                x = int(x.item() * 3999)
                y = int(y.item() * 3999)
                x = min(max(x, 1), 3999)
                y = min(max(y, 1), 3999)
                f.write(f"{x},{y}\n")

def analyze_feature_distributions(generators, data):
    samples = generate_samples(generators, data, num_samples=200)
    samples["human_data"] = data
    plt.ioff()
    fig_data = {}
    for model, model_name in zip(samples.keys(), image_titles):
        print(f"{model}")
        aggregated_statistics = []
        for generated_sequence in samples[model]:
            actions = convert_to_action_sequence(generated_sequence)
            if actions == None:
                continue
            aggregated_statistics.append(actions)
        aggregated_statistics = list(map(list, zip(*aggregated_statistics)))
        aggregated_statistics = dict(zip(feature_keys, aggregated_statistics))
        fig_data[model] = aggregated_statistics
        print_examples(model_name, samples[model])

    for feature_key in feature_keys:
        plt.title(f"Distribution of {feature_key}")
        t = [fig_data[model][feature_key] for model in samples.keys()]
        num_distinct = len(set([item for sublist in t for item in sublist]))
        min_val = min([min(fig_data[model][feature_key]) for model in samples.keys()])
        max_val = max([max(fig_data[model][feature_key]) for model in samples.keys()]) + 1
        num_bins = math.ceil(num_distinct / 5)
        bin_width = (max_val - min_val) / num_bins
        bins = list(range(int(min_val), int(max_val + bin_width), int(bin_width + 1)))
        for model, label in zip(samples.keys(), image_titles):
            results = plt.hist(fig_data[model][feature_key], bins=bins,
                            weights=np.ones_like(fig_data[model][feature_key]) / len(fig_data[model][feature_key]), label=label, stacked=True)
        fig1 = plt.gcf()
        plt.legend()
        fig1.savefig(f"machine_learning/images/Histograms/{feature_key}.png")
        plt.close(fig1)







if __name__ == "__main__":
    raw_mvmts_train, raw_mvmts_dev, raw_mvmts_test = load_data()
    generators = [
        "machine_learning/models/gan_generator_9_1.3132_0.62862.mdl",
        "machine_learning/models/gan_generator_4_0.31326_1.6265.mdl",
        "machine_learning/models/trained_generator_56_0.0002965_{'input_size': 3, 'hidden_size': 96, 'dropout': 0.2, 'bidrectional': False, 'num_layers': 4}.mdl",
        "machine_learning/models/pretrained_generator_55_0.00017816_{'input_size': 3, 'hidden_size': 96, 'dropout': 0.2, 'bidrectional': False, 'num_layers': 4}_prioritize_xy.mdl",
    ]

    with torch.no_grad():
        analyze_feature_distributions(generators, raw_mvmts_train)