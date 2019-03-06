import os
import re

import torch
from config import PROJECT_ROOT


def save_model(rnn, model_name):
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(rnn, os.path.join(models_dir, model_name))


def load_model(path, device="cuda"):
    model = torch.load(path)
    return model.to(device)


def read_log(path):
    pattern = r"\D*(\d*)\D*(\d*.\d*)"
    epochs, losses = zip(*[re.findall(pattern, line)[0] for line in open(path, "r")])

    epochs = [int(epoch) for epoch in epochs]
    losses = [float(loss) for loss in losses]

    return epochs, losses
