import argparse
from parse_config import ConfigParser
import deepmvlm
from utils3d import Utils3D
import os


def main(config):
    directory = 'D:/Data/temp/'
    names = Utils3D.get_mesh_files_in_dir(directory)
    # file_name = 'D:/Data/temp/MartinStandard.obj'
    # out_name = 'D:/Data/temp/MartinStandard_landmarks.vtk'
    # out_name_text = 'D:/Data/temp/MartinStandard_landmarks.txt'
    # file_name = 'I:/Data/temp/MartinStandard.obj'
    # out_name = 'I:/Data/temp/MartinStandard_landmarks.vtk'
    # out_name_text = 'I:/Data/temp/MartinStandard_landmarks.txt'
    file_name = config.name
    name_lm_vtk = os.path.splitext(file_name)[0] + '_landmarks.vtk'
    name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
    print('Processing ', file_name)
    # file_name = 'I:/Data/temp/Pasha_guard_head.obj'
    # out_name = 'I:/Data/temp/Pasha_guard_head_landmarks.vtk'
    # out_name_text = 'I:/Data/temp/Pasha_guard_head_landmarks.txt'

    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict_one_file(file_name)
    dm.write_landmarks_as_vtk_points(landmarks, name_lm_vtk)
    dm.write_landmarks_as_text(landmarks, name_lm_txt)
    dm.visualise_mesh_and_landmarks(file_name, landmarks)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist or directory to be processed')

    global_config = ConfigParser(args)
    main(global_config)
