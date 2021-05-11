import pandas as pd
import time

def read_data(path):
  with open(path) as file:
    data = [line[:-1] for line in file.readlines()]
    return data
  return []

def read_src_trg(src_path, trg_path, partition=5000):
    src = read_data(src_path)
    src = src[:len(src)//partition]

    trg = read_data(trg_path)
    trg = src[:len(trg)//partition]

    return src, trg

def write_to_tsv(path, src, trg):
    df = pd.DataFrame({
        "src": src,
        "trg": trg
        })
    df.to_csv(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs