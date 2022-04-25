import argparse
from parse_config import ConfigParser
import deepmvlm
from utils3d import Utils3D
import os


def process_one_file(config, file_name):
    print('Processing ', file_name)
    name_lm_vtk = os.path.splitext(file_name)[0] + '_landmarks.vtk'
    name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict_one_file(file_name)
    dm.write_landmarks_as_vtk_points(landmarks, name_lm_vtk)
    dm.write_landmarks_as_text(landmarks, name_lm_txt)
    dm.visualise_mesh_and_landmarks(file_name, landmarks)


def process_file_list(config, file_name):
    print('Processing filelist ', file_name)
    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) > 4:
                names.append(line)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        print('Processing ', file_name)
        name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(file_name)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def process_files_in_dir(config, dir_name):
    print('Processing files in  ', dir_name)
    names = Utils3D.get_mesh_files_in_dir(dir_name)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        print('Processing ', file_name)
        name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(file_name)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)


def main(config):
    name = str(config.name)
    if name.lower().endswith(('.obj', '.wrl', '.vtk', '.vtp', '.ply', '.stl')) and os.path.isfile(name):
        process_one_file(config, name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name)
    elif os.path.isdir(name):
        process_files_in_dir(config, name)
    else:
        print('Cannot process (not a mesh file, a filelist (.txt) or a directory)', name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')

    global_config = ConfigParser(args)
    main(global_config)
