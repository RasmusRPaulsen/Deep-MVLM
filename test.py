import argparse
import datetime
import time

import torch
import model.model as module_arch
from parse_config import ConfigParser
from utils3d import Utils3D
from utils3d import Render3D
from prediction import Predict2D
import os
import numpy as np
from scipy.spatial import distance
import math
import vtk
import sys


def get_working_device(config):
    device = torch.device('cpu')
    if config['n_gpu'] >= 1 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 3:
        device = torch.device('cuda')

    return device


def get_device_and_load_model(config):
    logger = config.get_logger('test')

    logger.debug('Initialising model')
    model = config.initialize('arch', module_arch)
    # logger.info(model)

    if config.resume is None:
        logger.error('Expecting model to be specified using the --r flag')
        return None, None

    check_point_name = str(config.resume)

    logger.info('Loading checkpoint: {}'.format(check_point_name))

    device = get_working_device(config)
    checkpoint = torch.load(check_point_name, map_location=device)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1 and device == torch.device('cuda'):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    epochs = checkpoint['epoch']
    logger.debug('Model was trained for ' + str(epochs) + ' epochs')

    # prepare model for predicting
    model = model.to(device)
    model.eval()

    return device, model


def predict_one_subject(config, file_name):
    device, model = get_device_and_load_model(config)

    render_3d = Render3D(config)
    image_stack, transform_stack = render_3d.render_3d_file(file_name)

    predict_2d = Predict2D(config, model, device)
    heatmap_maxima = predict_2d.predict_heatmaps_from_images(image_stack)

    u3d = Utils3D(config)
    u3d.heatmap_maxima = heatmap_maxima
    # u3d.read_heatmap_maxima()
    # u3d.read_3d_transformations()
    u3d.transformations_3d = transform_stack
    u3d.compute_lines_from_heatmap_maxima()
    # u3d.visualise_one_landmark_lines(40, 'saved/temp/DeepMVLM_DTU3D/0904_104414')
    u3d.compute_all_landmarks_from_view_lines()
    u3d.project_landmarks_to_surface(file_name)
    # u3d.write_landmarks_as_vtk_points()
    return u3d.landmarks


def read_3d_landmarks(file_name):
    lms = []
    with open(file_name) as f:
        for line in f:
            line = line.strip("/n")
            x, y, z = np.double(line.split(" "))
            lms.append((x, y, z))
    return lms


def write_landmark_accuracy(gt_lm, pred_lm, file):
    if len(gt_lm) != len(pred_lm):
        print('Number of gt landmarks ', len(gt_lm), ' does not match number of predicted lm ', len(pred_lm))
        return None

    sum_dist = 0
    for idx in range(len(gt_lm)):
        gt_p = gt_lm[idx]
        pr_p = pred_lm[idx]
        dst = distance.euclidean(gt_p, pr_p)
        sum_dist = sum_dist + dst
        file.write(str(dst))
        if idx != len(gt_lm):
            file.write(', ')
    file.write('\n')
    print('Average landmark error ', sum_dist / len(gt_lm))


# TODO Use render3d version
def get_landmark_bounds(lms):
    x_min = lms[0][0]
    x_max = x_min
    y_min = lms[0][1]
    y_max = y_min
    z_min = lms[0][2]
    z_max = z_min

    for lm in lms:
        x = lm[0]
        y = lm[1]
        z = lm[2]
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
        z_min = min(z_min, z)
        z_max = max(z_max, z)

    return x_min, x_max, y_min, y_max, z_min, z_max


# TODO Use render3d version
def get_landmarks_bounding_box_diagonal_length(lms):
    x_min, x_max, y_min, y_max, z_min, z_max = get_landmark_bounds(lms)

    # Diagonal length
    diag_len = math.sqrt((x_max-x_min) * (x_max-x_min) + (y_max-y_min) * (y_max-y_min) + (z_max-z_min) * (z_max-z_min))
    return diag_len


# TODO Move to render3d or utils3D
def visualise_landmarks_as_spheres_with_accuracy(gt_lm, pred_lm, file_out):
    diag_len = get_landmarks_bounding_box_diagonal_length(gt_lm)
    # sphere radius is 10% of bounding box diagonal
    sphere_size = diag_len * 0.010

    append = vtk.vtkAppendPolyData()
    for idx in range(len(gt_lm)):
        gt_p = gt_lm[idx]
        pr_p = pred_lm[idx]
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(pr_p)
        sphere.SetRadius(sphere_size)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)
        sphere.Update()
        scalars.SetNumberOfValues(sphere.GetOutput().GetNumberOfPoints())

        dst = distance.euclidean(gt_p, pr_p)

        for s in range(sphere.GetOutput().GetNumberOfPoints()):
            scalars.SetValue(s, dst)

        sphere.GetOutput().GetPointData().SetScalars(scalars)
        append.AddInputData(sphere.GetOutput())
        del sphere
        del scalars

    append.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(append.GetOutput())
    writer.SetFileName(file_out)
    writer.Write()

    del writer
    del append


def write_lm_names_to_result_file(res_f):
    lm_name_file = os.path.dirname(sys.argv[0]) + "/docs/DTU-3D_landmark_info.txt"
    lm_names = []
    with open(lm_name_file) as f:
        for line in f:
            line = line.strip("/n")
            line = line.strip("\n")
            line = line.rstrip()
            t = np.array(line.split(';'))
            lm_names.append(t)
            # clean_name = os.path.splitext(line)[0]
            #if len(clean_name) > 0:
                #files.append(clean_name)
    for lm in lm_names:
        idx = lm[0]
        res_f.write(', ' + str(idx))
    res_f.write('\n')
    for lm in lm_names:
        name = lm[1]
        res_f.write(', ' + name)
    res_f.write('\n')
    res_f.flush()



def test_on_bu_3d_fe(config):
    test_set_file = config['data_loader']['args']['data_dir'] + '/dataset_test.txt'
    # test_set_file = config['data_loader']['args']['data_dir'] + '/face_dataset_debug.txt'
    result_file = config.temp_dir / 'results.csv'

    device, model = get_device_and_load_model(config)

    files = []
    with open(test_set_file) as f:
        for line in f:
            line = line.strip("/n")
            line = line.strip("\n")
            clean_name = os.path.splitext(line)[0]
            if len(clean_name) > 0:
                files.append(clean_name)
    print('Read', len(files), 'files to run test on')

    bu_3dfe_dir = config['preparedata']['raw_data_dir']

    idx = 0
    res_f = open(result_file, "w")
    start_time = time.time()
    for f_name in files:
        lm_name = bu_3dfe_dir + f_name + '_RAW_84_LMS.txt'
        wrl_name = bu_3dfe_dir + f_name + '_RAW.wrl'
        # bmp_name = bu_3dfe_dir + f_name + '_F3D.bmp'

        gt_lms = read_3d_landmarks(lm_name)
        if os.path.isfile(wrl_name):
            print('Computing file ', idx, ' of ', len(files))

            render_3d = Render3D(config)
            image_stack, transform_stack = render_3d.render_3d_file(wrl_name)

            predict_2d = Predict2D(config, model, device)
            heatmap_maxima = predict_2d.predict_heatmaps_from_images(image_stack)

            print('Computing 3D landmarks')
            u3d = Utils3D(config)
            u3d.heatmap_maxima = heatmap_maxima
            u3d.transformations_3d = transform_stack
            u3d.compute_lines_from_heatmap_maxima()
            # u3d.visualise_one_landmark_lines(83)
            # u3d.visualise_one_landmark_lines(26)
            u3d.compute_all_landmarks_from_view_lines()
            u3d.project_landmarks_to_surface(wrl_name)
            pred_lms = u3d.landmarks

            res_f.write(f_name + ', ')
            write_landmark_accuracy(gt_lms, pred_lms, res_f)
            res_f.flush()

            base_name = os.path.basename(f_name)
            sphere_file = config.temp_dir / (base_name + '_landmarkAccuracy.vtk')
            visualise_landmarks_as_spheres_with_accuracy(gt_lms, pred_lms, str(sphere_file))
            idx = idx + 1
            time_per_test = (time.time()-start_time) / idx
            time_left = (len(files) - idx) * time_per_test
            print('Time left in test: ', str(datetime.timedelta(seconds=time_left)))
        else:
            print('File', wrl_name, ' does not exists')


def main(config):
    test_on_bu_3d_fe(config)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    cfg_global = ConfigParser(args)
    main(cfg_global)
