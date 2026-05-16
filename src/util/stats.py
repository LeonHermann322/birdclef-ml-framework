import pandas as pd 
import numpy as np 
import ast
import librosa 
import os 

def get_audio_folder_stats(audio_folder_path):
    num_files = 0
    audio_lengths = []
    file_names = []
    
    #recursively walk through the audio folder and its subfolders 
    for root, dirs, files in os.walk(audio_folder_path):
        for file in files:
            if file.endswith(".ogg"):
                num_files += 1
                audio_path = os.path.join(root, file)
                audio_length = librosa.get_duration(path=audio_path)
                if audio_length > 600:
                    print("Skipping Audio file", file, "has length", audio_length, "seconds")
                else:
                    audio_lengths.append(audio_length)
                    file_names.append(file)

    print("Max length audio file", file_names[np.argmax(audio_lengths)], "with length", np.max(audio_lengths), "seconds")
    return num_files, audio_lengths

def get_label_distribution_count(df, taxonomy_df):
    #assert all filenames are unique 
    assert df["filename"].nunique() == len(df), "Filenames are not unique in the dataframe"
    assert "primary_label" in df.columns, "primary_label column is missing in the dataframe"
    
    #get label distribution 
    df_labels_count = {}
    for row in df.itertuples():
        labels = row.primary_label
        labels = labels.split(";")  # Split the string into a list of labels
        try: 
            secondary_labels = row.secondary_labels
            if pd.notna(secondary_labels):
                #get as list 
                secondary_labels = ast.literal_eval(secondary_labels)
                labels.extend(secondary_labels)  # Add secondary labels to the list of labels
        except:
            pass  # If secondary_labels column does not exist, ignore it

        for label in labels:
            maped_real_name = taxonomy_df[taxonomy_df["primary_label"] == label]["common_name"].values[0]
            if maped_real_name in df_labels_count:
                df_labels_count[maped_real_name] += 1
            else:
                df_labels_count[maped_real_name] = 1
    return df_labels_count

def get_label_distribution_count_for_train_audio(train_audio_df,taxonomy_df):
    return get_label_distribution_count(train_audio_df, taxonomy_df)

def get_label_distribution_count_for_soundscapes(sound_scapes_labels_df, taxonomy_df):
    sound_scapes_labels_df_file_fused = fuse_soundscapes_rows_per_file(sound_scapes_labels_df)
    return get_label_distribution_count(sound_scapes_labels_df_file_fused, taxonomy_df)

def fuse_soundscapes_rows_per_file(df):
    fused_dict = {}
    for row in df.itertuples():
        filename = row.filename
        labels = row.primary_label.split(";")  # Split the string into a list of labels

        if filename in fused_dict:
            fused_dict[filename].extend(labels)
        else:
            fused_dict[filename] = labels
    
    #remove duplicates from labels list for each file
    for key, value in fused_dict.items():
        fused_dict[key] = ";".join(list(set(value)))
    
    result_df = pd.DataFrame(list(fused_dict.items()), columns=["filename", "primary_label"])
    return result_df

def get_perch_labels():
    perch_labels = pd.read_csv("./data/birdclef_dataset/perch_model_class_mapping.csv")
    print("Number of species in Perch model:", len(perch_labels))

    perch_labels["scientific_name"] = perch_labels["inat2024_fsd50k"].str.lower()
    perch_labels = perch_labels["scientific_name"].tolist()
    return perch_labels