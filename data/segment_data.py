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
# !python data/segment_data.py --data-csv data/data.csv --data-path transcription_data_set

def get_args():
    """
    Get arguments from command line
    @return: argument list- csv path and the path for the data itself
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', required=True, type=str, help='path to data csv')
    parser.add_argument('--data-path', required=True, type=str, help='path to poses data')
    return parser.parse_args()


def create_segment_database(args):
    """
    Create a csv file for cut pose file entries out of original data csv
    @param args: csv path and the path for the data itself
    """
    parent_path = os.path.dirname(args.data_csv)
    segment_path = parent_path + "/data_segmentation.csv"
    aef_path = parent_path + "/temp.aef"
    with (open(args.data_csv, 'r', encoding='utf-8') as datafile,
          open(segment_path, 'w', newline='', encoding='utf-8') as segmentfile):
        # Create CSV reader and writer objects
        reader = csv.DictReader(datafile)
        writer = csv.writer(segmentfile)
        writer.writerow(next(reader))

        # Iterate over rows that are needed to be cut in data file and write it to the output file
        for line in reader:
            if line['start'] == "0":
                path = args.data_path + "/" + line['pose']

                cmd = ['pose_to_segments', f'--pose={path}', f'--elan={aef_path}']
                with subprocess.Popen(cmd,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE) as sub:
                    sub.wait()
                    eaf = pympi.Elan.Eaf(file_path=aef_path)

                    # Accessing annotations from the "SIGN" tier
                    sign_annotations = eaf.get_full_time_interval()

                    if sign_annotations[0] != sign_annotations[1]:
                        new_line = copy.deepcopy(line)
                        new_line['start'] = sign_annotations[0]
                        new_line['end'] = sign_annotations[1]
                        print(list(new_line.values()))
                        print(sign_annotations)
                        writer.writerow(list(new_line.values()))
    if os.path.exists(aef_path):
        os.remove(aef_path)

if __name__ == "__main__":
    arguments = get_args()
    create_segment_database(arguments)
