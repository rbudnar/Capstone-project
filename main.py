import pandas as pd

def generate_dataframe_from_csv(path):
    data = pd.read_csv(path)
    columns = (data.apply(lambda r: pd.Series(gen_image_paths(r)), axis=1)
        .stack()
        .rename("img_path")
        .reset_index(level=1, drop=True))
    data["sirna"] = data["sirna"].apply(lambda s: str(s))
    return data.join(columns).reset_index(drop=True)

def gen_image_paths(row):
    path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
    return [f"{path_root}_s{site}_w{image}.png" for site in range(1, 3) for image in range(1,7)]