import csv
import os


# write CSV file with rows [file_name,class]
# the resulting CSV file contains audio samples from the audio_dir directory 
# mixed with all the audio samples from the ESC-10 dataset.
def write_csv(audio_dir, csv_file):

    # open the file in the write mode
    f = open(os.path.join(os.getcwd(), csv_file), 'w')
    audio_path = os.path.join(os.getcwd(), audio_dir)
    env_sound_path = os.path.join(os.getcwd(), 'ESC-10')

    header = ['file_name', 'class_name']

    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(header)
    
    # write a row (file,class) for bark samples in the csv file
    for file in os.listdir(audio_path):
        data = [file, 'bark']
        writer.writerow(data)
        
    env_dirs = [dirname for dirname in os.listdir(env_sound_path) if not dirname.startswith("barks") and
                                                                    not dirname.startswith(".")]
    for env_dir in env_dirs:
        for file in os.listdir(os.path.join(env_sound_path, env_dir)):
            data = [file, env_dir]
            writer.writerow(data)
    
    # close the file
    f.close()