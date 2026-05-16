import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

def plot_audio_stats(num_files, audio_lengths, audio_folder_path):
    print("Number of audio files in {}: {}".format(audio_folder_path, num_files))
    print("Average audio length in seconds: {:.2f}".format(np.mean(audio_lengths))) 
    #plot histogram of audio lengths
    plt.hist(audio_lengths, bins=100)
    plt.xlabel("Audio Length (seconds)")
    plt.ylabel("Frequency")
    plt.show()
    plt.close()

    #plot whisker plot of audio lengths
    plt.figure()
    plt.boxplot(audio_lengths)
    plt.ylabel("Audio Length (seconds)")
    plt.show()
    plt.close()

#plot class distribution in taxonomy
def plot_family_distribution(taxonomy_df):
     family_counts = taxonomy_df["class_name"].value_counts()
     plt.figure(figsize=(10,5))
     family_counts.plot(kind="bar")
     plt.xlabel("Class Name")
     plt.ylabel("Count")
     plt.title("Class Distribution in Taxonomy")
     plt.xticks(rotation=90)
     plt.show()
     plt.close()

def plot_label_distribution_count(df_labels_count, title:str="" ):
    print("Number of animals labeled in train soundscapes labels: {}".format(len(df_labels_count)))
    #print("Label distribution in train soundscapes labels:")
    #for label, count in df_labels_count.items():
    #    print(f"{label}: {count}")
    #plot label distribution in train soundscapes labels
    plt.figure(figsize=(10,5))
    plt.bar(df_labels_count.keys(), df_labels_count.values())
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label Distribution" + title)
    plt.xticks(rotation=90)
    plt.show()
    plt.close()

def plot_date_distribution(df):
    df["date"] = df["filename"].apply(lambda x: x.split("_")[-2])
    date_counts = pd.to_datetime(df['date']).dt.date.value_counts().sort_index()

    plt.figure(figsize=(12, 5))
    plt.bar(date_counts.index, date_counts.values, width=15, color='#4C72B0', edgecolor='black')
    plt.title('Soundscapes Distribution by Date', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_time_distribution(df):
    df["time"] = df["filename"].apply(lambda x: x.split("_")[-1].split(".")[0])
    time_counts = pd.to_datetime(df['time'], format='%H%M%S').dt.time.value_counts().sort_index()

    plt.figure(figsize=(12, 5))

    plt.bar(time_counts.index.astype(str), time_counts.values, color='#DD8452', edgecolor='black')
    plt.title('Soundscapes Distribution by Time of Day', fontsize=14)
    plt.xlabel('Time (HH:MM:SS)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 


def plot_count_stats(count_dict, title):
    #get stats of distribution 
    print(pd.DataFrame(count_dict.values()).describe())

    #whisker plot of count dict values
    plt.figure()
    plt.boxplot(count_dict.values())
    plt.ylabel("Count")
    plt.title(title)
    plt.show()
    plt.close()

