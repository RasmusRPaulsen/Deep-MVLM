import argparse
import torch
import model.model as module_arch
from parse_config import ConfigParser
from utils3d import Utils3D
from utils3d import Render3D
from prediction import Predict2D


def get_working_device(config):
    device = torch.device('cpu')
    if config['n_gpu'] >= 1 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 3:
        device = torch.device('cuda')

    return device


def get_device_and_load_model(config):
    logger = config.get_logger('test')

    print('Initialising model')
    model = config.initialize('arch', module_arch)
    logger.info(model)

    print('Loading checkpoint')
    image_channels = config['data_loader']['args']['image_channels']
    if image_channels == "geometry":
        check_point_name = 'saved/trained/MVLMModel_DTU3D_geometry.pth'
    elif image_channels == "RGB":
        check_point_name = 'saved/trained/MVLMModel_DTU3D_RGB_07092019.pth'
    logger.info('Loading checkpoint: {}'.format(check_point_name))

    device = get_working_device(config)
    checkpoint = torch.load(check_point_name, map_location=device)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1 and device == torch.device('cuda'):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

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
    u3d.visualise_one_landmark_lines(65)
    u3d.compute_all_landmarks_from_view_lines()
    u3d.project_landmarks_to_surface(file_name)
    u3d.write_landmarks_as_vtk_points()


def main(config):
    # subject_name = 'I:/Data/temp/MartinStandard.obj'
    # subject_name = 'D:/Data/temp/MartinStandard.obj'
    # subject_name = 'D:/Data/temp/20130218092022_standard.obj'
    # subject_name = 'D:/Data/temp/20140508111926_standard.obj'
    # file_name = 'D:/Data/temp/20130715150920_standard.obj'
    # file_name = 'D:/Data/temp/20121105144354_standard.obj'
    # file_name = 'I:/Data/temp/V00171112671006_08-07-2011_ghjghj.obj' # Check right eye
    file_name = 'I:/Data/temp/V00171112599106_17-06-2011_elleve.obj'  # beard causing problem for LM 66 (65 with o index)
    predict_one_subject(config, file_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
