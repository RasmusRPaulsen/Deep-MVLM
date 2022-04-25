import math

import vtk
import numpy as np
import time
# from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy
import os

from utils3d import Utils3D


def no_transform():
    rx = 0
    ry = 0
    rz = 0
    scale = 1
    tx = 0
    ty = 0
    return rx, ry, rz, scale, tx, ty


class Render3D:
    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('Render3D')

    def random_transform(self):
        min_x = self.config['process_3d']['min_x_angle']
        max_x = self.config['process_3d']['max_x_angle']
        min_y = self.config['process_3d']['min_y_angle']
        max_y = self.config['process_3d']['max_y_angle']
        min_z = self.config['process_3d']['min_z_angle']
        max_z = self.config['process_3d']['max_z_angle']

        rx = np.double(np.random.randint(min_x, max_x, 1))
        ry = np.double(np.random.randint(min_y, max_y, 1))
        rz = np.double(np.random.randint(min_z, max_z, 1))
        # the following values are currently not used
        scale = np.double(np.random.uniform(1.4, 1.9, 1))
        tx = np.double(np.random.randint(-20, 20, 1))
        ty = np.double(np.random.randint(-20, 20, 1))

        return rx, ry, rz, scale, tx, ty

    # Generate nview 3D transformations and return them as a stack
    def generate_3d_transformations(self):
        n_views = self.config['data_loader']['args']['n_views']
        transform_stack = np.zeros((n_views, 6), dtype=np.float32)

        for idx in range(n_views):
            rx, ry, rz, s, tx, ty = self.random_transform()
            transform_stack[idx, :] = (rx, ry, rz, s, tx, ty)

        return transform_stack

    def compute_pre_transformation(self, file_name):
        translation = [0, 0, 0]
        if self.config['pre-align']['align_center_of_mass']:
            # hack to avoid how the objimporter deals with multiple polydata
            pd = Utils3D.multi_read_surface(file_name)
            if pd.GetNumberOfPoints() < 1:
                print('Could not read', file_name)
                return None

            vtk_cm = vtk.vtkCenterOfMass()
            vtk_cm.SetInputData(pd)
            vtk_cm.SetUseScalarsAsWeights(False)
            vtk_cm.Update()
            cm = vtk_cm.GetCenter()
            translation = [-cm[0], -cm[1], -cm[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = self.config['pre-align']['rot_x']
        ry = self.config['pre-align']['rot_y']
        rz = self.config['pre-align']['rot_z']
        # Scale is handling by doing magic with the view frustrum elsewhere
        # s = self.config['pre-align']['scale']

        # t.Scale(s, s, s)
        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()

        return t

    def render_3d_obj_rgb(self, transform_stack, file_name):
        write_image_files = self.config['process_3d']['write_renderings']
        off_screen_rendering = self.config['process_3d']['off_screen_rendering']
        n_views = self.config['data_loader']['args']['n_views']
        img_size = self.config['data_loader']['args']['image_size']
        win_size = img_size

        n_channels = 3
        image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)

        mtl_name = os.path.splitext(file_name)[0] + '.mtl'
        obj_dir = os.path.dirname(file_name)
        obj_in = vtk.vtkOBJImporter()
        obj_in.SetFileName(file_name)
        obj_in.SetFileNameMTL(mtl_name)
        obj_in.SetTexturePath(obj_dir)
        obj_in.Update()

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

        obj_in.SetRenderWindow(ren_win)
        obj_in.Update()

        props = vtk.vtkProperty()
        props.SetDiffuse(0)
        props.SetSpecular(0)
        props.SetAmbient(1)

        actors = ren.GetActors()
        actors.InitTraversal()
        actor = actors.GetNextItem()
        while actor:
            actor.SetProperty(props)
            actor = actors.GetNextItem()
        del props

        t_pre_trans = self.compute_pre_transformation(file_name)

        t = vtk.vtkTransform()
        t.Identity()
        t.Update()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(ren_win)
        writer_png = vtk.vtkPNGWriter()
        writer_png.SetInputConnection(w2if.GetOutputPort())

        start = time.time()
        # for idx in tqdm(range(n_views)):
        for idx in range(n_views):
            name_rendering = self.config.temp_dir / ('rendering' + str(idx) + '_RGB.png')

            rx, ry, rz, s, tx, ty = transform_stack[idx]
            # rx,ry,rz,s,tx,ty = no_transform() # debug
            # rx = -20
            # ry = 40
            # rz = 10

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Concatenate(t_pre_trans)
            t.Update()

            xmin = -150
            xmax = 150
            ymin = -150
            ymax = 150
            xlen = xmax - xmin
            ylen = ymax - ymin

            cx = 0
            cy = 0
            # extend_factor = 1.0
            s = self.config['pre-align']['scale']
            extend_factor = 1.0 / s
            # The side length of the view frustrum which is rectangular since we use a parallel projection
            side_length = max([xlen, ylen]) * extend_factor
            # zoom_factor = win_size / side_length

            ren.GetActiveCamera().SetParallelScale(side_length / 2)
            ren.GetActiveCamera().SetPosition(cx, cy, 500)
            ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
            ren.GetActiveCamera().SetViewUp(0, 1, 0)
            ren.GetActiveCamera().ApplyTransform(t.GetInverse())
            ren.ResetCameraClippingRange()  # This approach is not recommended when doing depth rendering

            ren_win.Render()

            if write_image_files:
                w2if.Modified()  # Needed here else only first rendering is put to file
                writer_png.SetFileName(str(name_rendering))
                writer_png.Write()
            else:
                w2if.Modified()  # Needed here else only first rendering is put to file
                w2if.Update()

            # add rendering to image stack
            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)

            image_stack[idx, :, :, :] = a[:, :, :]

        end = time.time()
        print("Pure RGB rendering time: " + str(end - start))

        del obj_in
        del writer_png, w2if
        del ren, ren_win, t
        return image_stack

    def apply_pre_transformation(self, pd):
        translation = [0, 0, 0]
        if self.config['pre-align']['align_center_of_mass']:
            vtk_cm = vtk.vtkCenterOfMass()
            vtk_cm.SetInputData(pd)
            vtk_cm.SetUseScalarsAsWeights(False)
            vtk_cm.Update()
            cm = vtk_cm.GetCenter()
            translation = [-cm[0], -cm[1], -cm[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = self.config['pre-align']['rot_x']
        ry = self.config['pre-align']['rot_y']
        rz = self.config['pre-align']['rot_z']
        s = self.config['pre-align']['scale']

        t.Scale(s, s, s)
        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()

        # Transform (assuming only one mesh)
        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        if self.config['pre-align']['write_pre_aligned']:
            name_out = str(self.config.temp_dir / ('pre_transform_mesh.vtk'))
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(trans.GetOutput())
            writer.SetFileName(name_out)
            writer.Write()

        return trans.GetOutput()

    def render_3d_multi_rgb_geometry_depth(self, transform_stack, file_name):
        write_image_files = self.config['process_3d']['write_renderings']
        off_screen_rendering = self.config['process_3d']['off_screen_rendering']
        n_views = self.config['data_loader']['args']['n_views']
        img_size = self.config['data_loader']['args']['image_size']
        win_size = img_size
        slack = 5

        start = time.time()
        self.logger.debug('Rendering')

        n_channels = 5  # 3 for RGB, 1 for depth and 1 for geometry
        image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)

        pd = Utils3D.multi_read_surface(file_name)
        if pd.GetNumberOfPoints() < 1:
            print('Could not read', file_name)
            return None

        pd = self.apply_pre_transformation(pd)

        texture_img = Utils3D.multi_read_texture(file_name)
        if texture_img is not None:
            pd.GetPointData().SetScalars(None)
            texture = vtk.vtkTexture()
            texture.SetInterpolate(1)
            texture.SetQualityTo32Bit()
            texture.SetInputData(texture_img)

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

        t = vtk.vtkTransform()
        t.Identity()
        t.Update()

        # Transform (assuming only one mesh)
        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(trans.GetOutput())

        actor_text = vtk.vtkActor()
        actor_text.SetMapper(mapper)
        if texture_img is not None:
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
            name_rgb = str(self.config.temp_dir / ('rendering' + str(view) + '_RGB.png'))
            name_depth = str(self.config.temp_dir / ('rendering' + str(view) + '_zbuffer.png'))
            name_geometry = str(self.config.temp_dir / ('rendering' + str(view) + '_geometry.png'))

            # print('Rendering ', name_rgb)
            rx, ry, rz, s, tx, ty = transform_stack[view]

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()
            trans.Update()

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
            # zoom_fac = win_size / side_length

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

            if write_image_files:
                w2if.Modified()  # Needed here else only first rendering is put to file
                writer_png.SetFileName(name_rgb)
                writer_png.Write()
            else:
                w2if.Modified()  # Needed here else only first rendering is put to file
                w2if.Update()

            # add rendering to image stack
            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)

            # get RGB data - 3 first channels
            image_stack[view, :, :, 0:3] = a[:, :, :]

            actor_text.SetVisibility(False)
            actor_geometry.SetVisibility(True)
            mapper.Modified()
            ren.Modified()  # force actors to have the correct visibility
            ren_win.Render()

            if write_image_files:
                w2if.Modified()  # Needed here else only first rendering is put to file
                writer_png.SetFileName(name_geometry)
                writer_png.Write()
            else:
                w2if.Modified()  # Needed here else only first rendering is put to file
                w2if.Update()

            # add rendering to image stack
            im = w2if.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)

            # get geometry data
            image_stack[view, :, :, 3:4] = a[:, :, 0:1]

            ren.Modified()  # force actors to have the correct visibility
            ren_win.Render()
            w2if.SetInputBufferTypeToZBuffer()
            w2if.Modified()

            if write_image_files:
                w2if.Modified()  # Needed here else only first rendering is put to file
                writer_png_2.SetFileName(name_depth)
                writer_png_2.Write()
            else:
                w2if.Modified()  # Needed here else only first rendering is put to file
                w2if.Update()

            scale.Update()
            im = scale.GetOutput()
            rows, cols, _ = im.GetDimensions()
            sc = im.GetPointData().GetScalars()
            a = vtk_to_numpy(sc)
            components = sc.GetNumberOfComponents()
            a = a.reshape(rows, cols, components)
            a = np.flipud(a)

            # get depth data
            image_stack[view, :, :, 4:5] = a[:, :, 0:1]

            actor_geometry.SetVisibility(False)
            actor_text.SetVisibility(True)
            ren.Modified()

        del writer_png_2, writer_png, ren_win, actor_geometry, actor_text, mapper, w2if, t, trans
        if texture_img is not None:
            del texture_img
            del texture
        # del texture_image
        end = time.time()
        self.logger.debug("File load and rendering time: " + str(end - start))

        return image_stack

    def render_3d_file(self, file_name):
        image_channels = self.config['data_loader']['args']['image_channels']
        file_type = (os.path.splitext(file_name)[1]).lower()

        image_stack = None
        transformation_stack = None
        n_views = self.config['data_loader']['args']['n_views']
        win_size = self.config['data_loader']['args']['image_size']

        if file_type == ".obj" and image_channels == "RGB":
            transformation_stack = self.generate_3d_transformations()
            image_stack = self.render_3d_obj_rgb(transformation_stack, file_name)
            image_stack = image_stack / 255
        elif file_type == ".obj" and image_channels == "RGB+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_rgb = self.render_3d_obj_rgb(transformation_stack, file_name)
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            # image_stack_depth = self.render_3d_obj_depth(transformation_stack, file_name)
            n_channels = 4
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_rgb / 255
            image_stack[:, :, :, 3:4] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl"]) and image_channels == "RGB":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            n_channels = 3
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_full[:, :, :, 0:3] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "geometry":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            n_channels = 1
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 3:4] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            n_channels = 1
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl"]) and image_channels == "RGB+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            n_channels = 4
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:3] = image_stack_full[:, :, :, 0:3] / 255
            image_stack[:, :, :, 3:4] = image_stack_full[:, :, :, 4:5] / 255
        elif (file_type in [".vtk", ".vtp", ".stl", ".ply", ".wrl", ".obj"]) and image_channels == "geometry+depth":
            transformation_stack = self.generate_3d_transformations()
            image_stack_full = self.render_3d_multi_rgb_geometry_depth(
                transformation_stack, file_name)
            n_channels = 2
            image_stack = np.zeros((n_views, win_size, win_size, n_channels), dtype=np.float32)
            image_stack[:, :, :, 0:1] = image_stack_full[:, :, :, 3:4] / 255
            image_stack[:, :, :, 1:2] = image_stack_full[:, :, :, 4:5] / 255
        else:
            print("Can not render filetype ", file_type, " using image_channels ", image_channels)

        return image_stack, transformation_stack

    @staticmethod
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

    @staticmethod
    def get_landmarks_bounding_box_diagonal_length(lms):
        x_min, x_max, y_min, y_max, z_min, z_max = Render3D.get_landmark_bounds(lms)

        diag_len = math.sqrt(
            (x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min) + (z_max - z_min) * (z_max - z_min))
        return diag_len

    @staticmethod
    def get_landmarks_as_spheres(lms):
        diag_len = Render3D.get_landmarks_bounding_box_diagonal_length(lms)
        # sphere radius is 0.8% of bounding box diagonal
        sphere_size = diag_len * 0.008

        append = vtk.vtkAppendPolyData()
        for idx in range(len(lms)):
            lm = lms[idx]
            # scalars = vtk.vtkDoubleArray()
            # scalars.SetNumberOfComponents(1)

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(lm)
            sphere.SetRadius(sphere_size)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()
            # scalars.SetNumberOfValues(sphere.GetOutput().GetNumberOfPoints())

            # for s in range(sphere.GetOutput().GetNumberOfPoints()):
            #    scalars.SetValue(s, dst)

            # sphere.GetOutput().GetPointData().SetScalars(scalars)
            append.AddInputData(sphere.GetOutput())
            del sphere
            # del scalars

        append.Update()
        return append.GetOutput()

    @staticmethod
    def visualise_mesh_and_landmarks(mesh_name, landmarks=None):
        file_type = os.path.splitext(mesh_name)[1]
        win_size = 512

        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)

        # Initialize RenderWindow
        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(ren)
        ren_win.SetSize(win_size, win_size)

        file_read = False
        if file_type == ".obj":
            mtl_name = os.path.splitext(mesh_name)[0] + '.mtl'
            # only do this for textured files
            if os.path.isfile(mtl_name):
                obj_dir = os.path.dirname(mesh_name)
                obj_in = vtk.vtkOBJImporter()
                obj_in.SetFileName(mesh_name)
                if os.path.isfile(mtl_name):
                    obj_in.SetFileNameMTL(mtl_name)
                    obj_in.SetTexturePath(obj_dir)
                obj_in.Update()

                obj_in.SetRenderWindow(ren_win)
                obj_in.Update()

                props = vtk.vtkProperty()
                props.SetColor(1, 1, 1)
                props.SetDiffuse(0)
                props.SetSpecular(0)
                props.SetAmbient(1)

                actors = ren.GetActors()
                actors.InitTraversal()
                actor = actors.GetNextItem()
                while actor:
                    actor.SetProperty(props)
                    actor = actors.GetNextItem()
                del props
                file_read = True

        if not file_read and file_type in [".vtk", ".stl", ".vtp", ".ply", ".wrl", ".obj"]:
            pd = Utils3D.multi_read_surface(mesh_name)
            if pd.GetNumberOfPoints() < 1:
                print('Could not read', mesh_name)
                return None

            texture_img = Utils3D.multi_read_texture(mesh_name)
            if texture_img is not None:
                pd.GetPointData().SetScalars(None)
                texture = vtk.vtkTexture()
                texture.SetInterpolate(1)
                texture.SetQualityTo32Bit()
                texture.SetInputData(texture_img)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)

            actor_text = vtk.vtkActor()
            actor_text.SetMapper(mapper)
            if texture_img is not None:
                actor_text.SetTexture(texture)
                actor_text.GetProperty().SetColor(1, 1, 1)
                actor_text.GetProperty().SetAmbient(1.0)
                actor_text.GetProperty().SetSpecular(0)
                actor_text.GetProperty().SetDiffuse(0)
            ren.AddActor(actor_text)

        if landmarks is not None:
            lm_pd = Render3D.get_landmarks_as_spheres(landmarks)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(lm_pd)

            actor_lm = vtk.vtkActor()
            actor_lm.SetMapper(mapper)
            actor_lm.GetProperty().SetColor(0, 0, 1)
            ren.AddActor(actor_lm)

        # ren.GetActiveCamera().SetPosition(0, 0, 1)
        # ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        # ren.GetActiveCamera().SetViewUp(0, 1, 0)
        # ren.GetActiveCamera().SetParallelProjection(1)

        iren = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        iren.SetRenderWindow(ren_win)

        ren_win.Render()
        iren.Start()
