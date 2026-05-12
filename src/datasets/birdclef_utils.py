import os
import pandas as pd

def create_random_val_split(yaml_config, val_fraction=0.2, random_seed=42):
    
    print("val_split.csv not found. Creating val_split.csv by splitting train_split.csv")
    #split train_split.csv randomly by 20% to create val_split.csv if val_split.csv doesn't exist
    train_df = pd.read_csv(yaml_config.base_data_dir + "/train_split.csv")
    val_df = train_df.sample(frac=val_fraction, random_state=random_seed)
    train_df = train_df.drop(val_df.index)
    train_df.to_csv(yaml_config.base_data_dir + "/train_split.csv", index=False)
    val_df.to_csv(yaml_config.base_data_dir + "/val_split.csv", index=False)

def load_splits(yaml_config):
    train_path = os.path.join(yaml_config.project_root_dir, "train_split.csv")
    val_path = os.path.join(yaml_config.project_root_dir, "val_split.csv")
    test_path = os.path.join(yaml_config.project_root_dir, "test_split.csv")

    if not os.path.exists(yaml_config.base_data_dir + "/val_split.csv"):
        create_random_val_split(yaml_config)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    #TODO: remove this, after adding the file split handle 
    train_df = train_df[~ train_df["filename"].str.contains("_train_start_")].reset_index(drop=True)
    val_df = val_df[~ val_df["filename"].str.contains("_train_start_")].reset_index(drop=True)
    test_df = test_df[~ test_df["filename"].str.contains("_test_start_")].reset_index(drop=True)

    return train_df, val_df, test_df
