import pandas as pd
import numpy as np

INPUTS = 6


def generate_dataframe_from_csv_vertical(path, inputs=INPUTS):
    data = pd.read_csv(path)
    columns = (data.apply(lambda r: pd.Series(gen_image_paths_vertical(r, inputs)), axis=1)
               .stack()
               .rename("img_path")
               .reset_index(level=1, drop=True))
    if "sirna" in data.columns:
        data["sirna"] = data["sirna"].apply(lambda s: str(s))
    return data.join(columns).reset_index(drop=True)


def gen_image_paths_vertical(row, inputs=INPUTS):
    path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
    return [f"{path_root}_s{site}_w{image}.png" for site in range(1, 3) for image in range(1, 1+inputs)]


def gen_image_paths_horizontal(row):
    path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
    return [f"{path_root}_s{site}" for site in range(1, 3)]


def generate_dataframe_from_csv_horizontal(path, inputs=INPUTS, root_name_only=False):
    data = pd.read_csv(path)
    columns = (data.apply(lambda r: pd.Series(gen_image_paths_horizontal(r)), axis=1)
               .stack()
               .rename("img_path_root")
               .reset_index(level=1, drop=True))
    if "sirna" in data.columns:
        data["sirna"] = data["sirna"].apply(lambda s: str(s))

    data = data.join(columns).reset_index(drop=True)

    if not root_name_only:
        for i in range(1, 1+inputs):
            data[f"img_path_{i}"] = data.apply(
                lambda row: f"{row['img_path']}_w{i}.png", axis=1)
    return data


def get_model_inputs(df):
    trainY = df["sirna"]
    im_paths = df["img_path"].apply(
        lambda r: [f"{r}_w{image}.png" for image in range(1, 7)])
    splits = np.hsplit(np.stack(np.array(im_paths)), 6)

    images = [np.hstack(s) for s in splits]

    return (images, trainY)
