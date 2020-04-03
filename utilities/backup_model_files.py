from glob import glob
from os import path
from shutil import copyfile
import time
from datetime import datetime
import argparse


DELAY = 60*10

def is_same_size(file1, file2):
    size1 = path.getsize(file1)
    size2 = path.getsize(file2)
    return size1==size2

def backup_new_files(source_folder, destination_folder):
    search_pattern_source = path.join(source_folder, 'model.ckpt-*')
    search_pattern_destination = path.join(destination_folder, 'model.ckpt-*')

    source_file_list = glob(search_pattern_source)
    destination_file_list = glob(search_pattern_destination)

    source_file_names_list = [path.basename(file) for file in source_file_list]
    destination_file_names_list = [path.basename(file) for file in destination_file_list]

    for source_file_name in source_file_names_list:
        if source_file_name in destination_file_names_list and \
                is_same_size(file1=path.join(source_folder,source_file_name), file2=path.join(destination_folder,source_file_name)):
                    continue
        try:
            source_path = path.join(source_folder, source_file_name)
            destination_path = path.join(destination_folder, source_file_name)
            print(f'Copying file FROM: {source_path} TO: {destination_path} ')
            copyfile(source_path, destination_path)
        except Exception as e:
            print(f'Exception raised. Error message: {e}')

def run(source_folder, destination_folder):


    while True:
        time.sleep(5)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'{current_time} >> Checking new files')
        backup_new_files(source_folder, destination_folder)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy new files every 10min from source to destination path')
    parser.add_argument('-src', type=str, help='Source folder path')
    parser.add_argument('-dst', type=str, help='Destination folder path')

    args = parser.parse_args()
    run(source_folder=args.src,
        destination_folder=args.dst)