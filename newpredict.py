import argparse
from parse_config import ConfigParser
import deepmvlm


def main(config):
    # file_name = 'D:/Data/temp/MartinStandard.obj'
    # out_name = 'D:/Data/temp/MartinStandard_landmarks.vtk'
    # out_name_text = 'D:/Data/temp/MartinStandard_landmarks.txt'
    # file_name = 'I:/Data/temp/MartinStandard.obj'
    # out_name = 'I:/Data/temp/MartinStandard_landmarks.vtk'
    # out_name_text = 'I:/Data/temp/MartinStandard_landmarks.txt'
    file_name = 'I:/Data/temp/Pasha_guard_head.obj'
    out_name = 'I:/Data/temp/Pasha_guard_head_landmarks.vtk'
    out_name_text = 'I:/Data/temp/Pasha_guard_head_landmarks.txt'

    dm = deepmvlm.DeepMVLM(config)
    landmarks = dm.predict_one_file(file_name)
    dm.write_landmarks_as_vtk_points(landmarks, out_name)
    dm.write_landmarks_as_text(landmarks, out_name_text)
    dm.visualise_mesh_and_landmarks(file_name, landmarks)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    global_config = ConfigParser(args)
    main(global_config)
