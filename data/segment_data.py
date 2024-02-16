import argparse
import csv
import os
import subprocess
import pympi
import copy


# to run in colab:
# %%capture
# !pip install git+https://github.com/sign-language-processing/segmentation
# !pip install pympi-ling
# !python data/segment_data.py --data-csv data/data.csv --data-path transcription_data_set

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', required=True, type=str, help='path to data csv')
    parser.add_argument('--data-path', required=True, type=str, help='path to poses data')
    return parser.parse_args()

def create_segment_database(args):
    poses_path = args.data_path
    data_path = args.data_csv
    parent_path = os.path.dirname(args.data_csv)
    segment_path = parent_path + "/data_segmentation.csv"
    aef_path = parent_path + "/temp.aef"
    with (open(data_path, 'r', encoding='utf-8') as datafile,
          open(segment_path, 'w', newline='') as segmentfile,
          open(f'{aef_path}', 'w', newline='') as tempeaf):
        # Create CSV reader and writer objects
        reader = csv.DictReader(datafile)
        writer = csv.writer(segmentfile)
        headers = next(reader)
        writer.writerow(headers)

        # Iterate over each row that is needed to be cut in data file and write it to the output file
        for line in reader:
            if line['start'] == "0":
                path = poses_path + "/" + line['pose']

                sub = subprocess.Popen(['pose_to_segments',
                                        f'--pose={path}', f'--elan={aef_path}'])
                sub.wait()

                eaf = pympi.Elan.Eaf(file_path=aef_path)

                # Accessing annotations from the "SIGN" tier
                sign_annotations = eaf.get_full_time_interval()

                print(sign_annotations)

                if sign_annotations[0] != sign_annotations[1]:
                    new_line = copy.deepcopy(line)
                    new_line['start'] = sign_annotations[0]
                    new_line['end'] = sign_annotations[1]
                    print(list(new_line.values()))
                    writer.writerow(list(new_line.values()))

    if os.path.exists(aef_path):
        os.remove(aef_path)

if __name__ == "__main__":
    args = get_args()
    create_segment_database(args)
