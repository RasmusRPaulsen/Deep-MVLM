import argparse
from parse_config import ConfigParser
import os
import socket
import vtk
import numpy as np


def create_lock_file(name):
    f = open(name, "w")
    f.write(socket.gethostname())
    f.close()


def delete_lock_file(name):
    if os.path.exists(name):
        os.remove(name)


def random_transform(config):
    min_x = config['process_3d']['min_x_angle']
    max_x = config['process_3d']['max_x_angle']
    min_y = config['process_3d']['min_y_angle']
    max_y = config['process_3d']['max_y_angle']
    min_z = config['process_3d']['min_z_angle']
    max_z = config['process_3d']['max_z_angle']

    rx = np.double(np.random.randint(min_x, max_x, 1))
    ry = np.double(np.random.randint(min_y, max_y, 1))
    rz = np.double(np.random.randint(min_z, max_z, 1))
    # TODO the following values are not used
    scale = np.double(np.random.uniform(1.4, 1.9, 1))
    tx = np.double(np.random.randint(-20, 20, 1))
    ty = np.double(np.random.randint(-20, 20, 1))

    return rx, ry, rz, scale, tx, ty


def process_file_bu_3dfe(config, file_name, output_dir):
    bu_3dfe_dir = config['preparedata']['raw_data_dir']
    base_name = os.path.basename(file_name)
    name_pd = bu_3dfe_dir + file_name + '_RAW.wrl'
    name_bmp = bu_3dfe_dir + file_name + '_F3D.bmp'
    name_lm = bu_3dfe_dir + file_name + '_RAW_84_LMS.txt'

    name_path = os.path.dirname(file_name)
    o_dir_image = output_dir + '/images/' + name_path + '/'
    o_dir_lm = output_dir + '/2D LM/' + name_path + '/'
    if not os.path.exists(o_dir_image):
        os.makedirs(o_dir_image)
    if not os.path.exists(o_dir_lm):
        os.makedirs(o_dir_lm)

    lock_file = o_dir_image + base_name + '.lock'

    if not os.path.isfile(name_pd):
        print(name_pd, ' could not read')
        return False
    if not os.path.isfile(name_bmp):
        print(name_bmp, ' could not read')
        return False
    if not os.path.isfile(name_lm):
        print(name_lm, ' could not read')
        return False
    if os.path.isfile(lock_file):
        print(file_name, ' is locked - skipping')
        return True
    create_lock_file(lock_file)
    print('Rendering ', file_name)

    win_size = config['data_loader']['args']['image_size']
    off_screen_rendering = config['preparedata']['off_screen_rendering']
    n_views = config['data_loader']['args']['n_views']
    slack = 5

    # Load Landmarks
    points = vtk.vtkPoints()
    lms = vtk.vtkPolyData()

    with open(name_lm) as f:
        for line in f:
            line = line.strip("/n")
            x, y, z = np.double(line.split(" "))
            points.InsertNextPoint(x, y, z)
    lms.SetPoints(points)
    del points

    vrmlin = vtk.vtkVRMLImporter()
    vrmlin.SetFileName(name_pd)
    vrmlin.Update()

    pd = vrmlin.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
    pd.GetPointData().SetScalars(None)

    # Load texture
    texture_image = vtk.vtkBMPReader()
    texture_image.SetFileName(name_bmp)
    texture_image.Update()

    texture = vtk.vtkTexture()
    texture.SetInterpolate(1)
    texture.SetQualityTo32Bit()
    texture.SetInputConnection(texture_image.GetOutputPort())

    # Initialize Camera
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.GetActiveCamera().SetPosition(0, 0, 1)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 1, 0)
    ren.GetActiveCamera().SetParallelProjection(1)

    # Initialize RenderWindow
    ren_win = vtk.vtkRenderWindow()
    ren_win.AddRenderer(ren)
    ren_win.SetSize(win_size, win_size)
    ren_win.SetOffScreenRendering(off_screen_rendering)

    # Initialize Transform
    t = vtk.vtkTransform()
    t.Identity()
    t.Update()

    # Transform (assuming only one mesh)
    trans = vtk.vtkTransformPolyDataFilter()
    trans.SetInputData(pd)
    trans.SetTransform(t)
    trans.Update()

    trans_lm = vtk.vtkTransformPolyDataFilter()
    trans_lm.SetInputData(lms)
    trans_lm.SetTransform(t)
    trans_lm.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(trans.GetOutput())

    actor_text = vtk.vtkActor()
    actor_text.SetMapper(mapper)
    actor_text.SetTexture(texture)
    actor_text.GetProperty().SetColor(1, 1, 1)
    actor_text.GetProperty().SetAmbient(1.0)
    actor_text.GetProperty().SetSpecular(0)
    actor_text.GetProperty().SetDiffuse(0)
    ren.AddActor(actor_text)

    actor_geometry = vtk.vtkActor()
    actor_geometry.SetMapper(mapper)
    ren.AddActor(actor_geometry)

    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(ren_win)
    writer_png = vtk.vtkPNGWriter()
    writer_png.SetInputConnection(w2if.GetOutputPort())

    scale = vtk.vtkImageShiftScale()
    scale.SetOutputScalarTypeToUnsignedChar()
    scale.SetInputConnection(w2if.GetOutputPort())
    scale.SetShift(0)
    scale.SetScale(-255)

    writer_png_2 = vtk.vtkPNGWriter()
    writer_png_2.SetInputConnection(scale.GetOutputPort())

    for view in range(n_views):
        name_rgb = o_dir_image + base_name + '_' + str(view) + '.png'
        name_geometry = o_dir_image + base_name + '_' + str(view) + '_geometry.png'
        name_depth = o_dir_image + base_name + '_' + str(view) + '_zbuffer.png'
        name_2dlm = o_dir_lm + base_name + '_' + str(view) + '.txt'

        if not os.path.isfile(name_rgb):
            # print('Rendering ', name_rgb)
            rx, ry, rz, s, tx, ty = random_transform(config)

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()
            trans.Update()
            trans_lm.Update()

            xmin = -150
            xmax = 150
            ymin = -150
            ymax = 150
            zmin = trans.GetOutput().GetBounds()[4]
            zmax = trans.GetOutput().GetBounds()[5]
            xlen = xmax - xmin
            ylen = ymax - ymin

            cx = 0
            cy = 0
            extend_factor = 1.0
            side_length = max([xlen, ylen]) * extend_factor
            zoom_fac = win_size / side_length

            ren.GetActiveCamera().SetParallelScale(side_length / 2)
            ren.GetActiveCamera().SetPosition(cx, cy, 500)
            ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
            ren.GetActiveCamera().SetClippingRange(500 - zmax - slack, 500 - zmin + slack)

            # Save textured image
            w2if.SetInputBufferTypeToRGB()

            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            mapper.Modified()
            ren.Modified()  # force actors to have the correct visibility
            ren_win.Render()

            w2if.Modified()  # Needed here else only first rendering is put to file
            writer_png.SetFileName(name_rgb)
            writer_png.Write()

            actor_text.SetVisibility(False)
            actor_geometry.SetVisibility(True)
            mapper.Modified()
            ren.Modified()  # force actors to have the correct visibility
            ren_win.Render()

            w2if.Modified()  # Needed here else only first rendering is put to file
            writer_png.SetFileName(name_geometry)
            writer_png.Write()

            ren.Modified()  # force actors to have the correct visibility
            ren_win.Render()
            w2if.SetInputBufferTypeToZBuffer()
            w2if.Modified()

            writer_png_2.SetFileName(name_depth)
            writer_png_2.Write()
            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            ren.Modified()

            # Write transformed landmarks
            f = open(name_2dlm, 'w')

            t_lm = trans_lm.GetOutput()
            for i in range(t_lm.GetNumberOfPoints()):
                x_pos = t_lm.GetPoint(i)[0]
                y_pos = t_lm.GetPoint(i)[1]

                x_pos_screen = (x_pos - cx) * zoom_fac + win_size / 2
                y_pos_screen = -(y_pos - cy) * zoom_fac + win_size / 2

                line = str(x_pos_screen) + ' ' + str(y_pos_screen) + '\n'
                f.write(line)
            f.close()

    del writer_png_2, writer_png, ren_win, actor_geometry, actor_text, mapper, w2if, t, trans, vrmlin, texture
    del texture_image
    del lms, trans_lm

    delete_lock_file(lock_file)


def split_data_into_train_and_test(base_file_names, output_dir):
    train_file = output_dir + "dataset_train.txt"
    test_file = output_dir + "dataset_test.txt"
    # F0001-F0046 + M0001-M0034 are used for training
    # F0047-F0056 + M0035-M0054 are used for testing
    f1 = open(train_file, 'w')
    f2 = open(test_file, 'w')

    new_set = []

    for name in base_file_names:
        n1 = os.path.dirname(name)
        if n1.find('F') > -1:
            n1 = n1.replace('F', '')
            num = int(n1)
            if num < 47:
                new_set.append(name)
                f1.write(name + '\n')
            else:
                f2.write(name + '\n')
        if n1.find('M') > -1:
            n1 = n1.replace('M', '')
            num = int(n1)
            if num < 35:
                new_set.append(name)
                f1.write(name + '\n')
            else:
                f2.write(name + '\n')
    f1.close()
    f2.close()
    return new_set


def prepare_bu_3dfe_data(config):
    print('Preparing BU-3DFE data')
    file_id_list = config['preparedata']['raw_data_dir'] + 'BU_3DFE_base_filelist_noproblems.txt'
    output_dir = config['preparedata']['processed_data_dir']
    image_out_dir = output_dir + 'images/'
    lm_out_dir = output_dir + '2D LM/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(image_out_dir):
        os.makedirs(image_out_dir)
    if not os.path.exists(lm_out_dir):
        os.makedirs(lm_out_dir)

    base_file_names = []
    with open(file_id_list) as f:
        for line in f:
            line = line.strip("/n")
            line = line.strip("\n")
            if len(line) > 0:
                base_file_names.append(line)
    print('Read ', len(base_file_names), ' file ids')

    base_file_names = split_data_into_train_and_test(base_file_names, output_dir)

    print('Processing ', len(base_file_names), ' file ids for training')

    for base_name in base_file_names:
        print('Processing ', base_name)
        process_file_bu_3dfe(config, base_name, output_dir)


def main(config):
    model_data_name = config['name']
    if model_data_name.find('BU_3DFE') > -1:
        prepare_bu_3dfe_data(config)


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
