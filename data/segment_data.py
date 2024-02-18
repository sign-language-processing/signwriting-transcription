"""
Script for creating a csv file from a current poses csv file, where each entry is a cut version of
one in the csv file
"""
import argparse
import csv
import os
import subprocess
import copy
import pympi


# to run in colab:
# %%capture
# !pip install git+https://github.com/sign-language-processing/segmentation
# !pip install pympi-ling
# !python data/segment_data.py --data-csv data/data.csv --poses-path transcription_data_set

def get_args():
    """
    Get arguments from command line
    @return: argument list- csv path and the path for the data itself
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', required=True, type=str, help='path to data csv')
    parser.add_argument('--poses-path', required=True, type=str, help='path to poses data')
    return parser.parse_args()


def create_segment_database(args):
    """
    Create a csv file for cut pose file entries out of original data csv
    @param args: csv path and the path for the data itself
    """
    parent_path = os.path.dirname(args.data_csv)
    segment_path = parent_path + "/data_segmentation.csv"
    eaf_path = parent_path + "/temp.eaf"
    with (open(args.data_csv, 'r', encoding='utf-8') as data_file,
          open(segment_path, 'w', newline='', encoding='utf-8') as segment_file):
        # Create CSV reader and writer objects
        reader = csv.DictReader(data_file)
        writer = csv.writer(segment_file)
        writer.writerow(next(reader))

        # Iterate over rows that are needed to be cut in data file and write it to the output file
        for line in reader:
            if line['start'] == "0":
                path = args.poses_path + "/" + line['pose']

                cmd = ['pose_to_segments', f'--pose={path}', f'--elan={eaf_path}']
                with subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) as sub:
                    sub.wait()
                    eaf = pympi.Elan.Eaf(file_path=eaf_path)

                    # Accessing annotations from the "SIGN" tier
                    sign_annotations = eaf.get_annotation_data_for_tier("SIGN")

                    if len(sign_annotations) != 0:
                        new_line = copy.deepcopy(line)
                        new_line['start'] = sign_annotations[0][0]
                        new_line['end'] = sign_annotations[len(sign_annotations)-1][1]
                        print(list(new_line.values()))
                        print(sign_annotations)
                        writer.writerow(list(new_line.values()))
    if os.path.exists(eaf_path):
        os.remove(eaf_path)

if __name__ == "__main__":
    arguments = get_args()
    create_segment_database(arguments)
