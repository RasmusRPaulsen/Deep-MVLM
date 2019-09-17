import argparse
import collections
import math
import copy

import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import matplotlib.pyplot as plt
import numpy as np
import random

import vtk
import numpy as np
import socket
import os, fnmatch
import re
import sys
import time
from tqdm import tqdm
from scipy import optimize
from utils3d import Utils3D
from utils3d import Render3D

import vtk.numpy_interface.dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy


def NoTransform():
    rx = 0
    ry = 0
    rz = 0

    scale = 1

    tx = 0
    ty = 0
    return rx, ry, rz, scale, tx, ty


def CreateRandomTransform():
    # These values are for the DTU-3D set
    rx = np.double(np.random.randint(-40, 40, 1))
    ry = np.double(np.random.randint(-80, 80, 1))
    rz = np.double(np.random.randint(-20, 20, 1))

    scale = np.double(np.random.uniform(1.4, 1.9, 1))

    tx = np.double(np.random.randint(-20, 20, 1))
    ty = np.double(np.random.randint(-20, 20, 1))

    # Kristines values for BU-3DFE
    # rx = np.double(np.random.randint(-90, 20, 1))
    # ry = np.double(np.random.randint(-60, 60, 1))
    # rz = np.double(np.random.randint(-60, 60, 1))

    # scale = np.double(np.random.uniform(1.4, 1.9, 1))

    # tx = np.double(np.random.randint(-20, 20, 1))
    # ty = np.double(np.random.randint(-20, 20, 1))
    return rx, ry, rz, scale, tx, ty


def CreateRenderings1Subject(subject_name, winsize, nLines):
    slack = 5
    write_image_files = True
    off_screen_rendering = False

    obj_in = vtk.vtkOBJReader()
    obj_in.SetFileName(subject_name)
    obj_in.Update()

    # pd = vrmlin.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
    # pd.GetPointData().SetScalars(None)

    # Load texture
    # textureImage = vtk.vtkBMPReader()
    # textureImage.SetFileName(bmp_path)
    # textureImage.Update()

    # texture = vtk.vtkTexture()
    # texture.SetInterpolate(1)
    # texture.SetQualityTo32Bit()
    # texture.SetInputConnection(textureImage.GetOutputPort())

    # del textureImage

    ############## DO THINGS:    #################
    # Clean data
    # clean = vtk.vtkCleanPolyData()
    # clean.SetInputData(pd)
    # clean.Update()

    for idx in range(0, nLines):
        print(idx)
        # Initialize Camera
        ren = vtk.vtkRenderer()
        ren.SetBackground(1, 1, 1)
        ren.GetActiveCamera().SetPosition(0, 0, 1)
        ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        ren.GetActiveCamera().SetViewUp(0, 1, 0)
        ren.GetActiveCamera().SetParallelProjection(1)

        # Initialize RenderWindow
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(winsize, winsize)
        renWin.SetOffScreenRendering(off_screen_rendering)

        # Initialize Transform
        T = vtk.vtkTransform()
        T.Identity()
        T.Update()

        ############## Assuming one mesh ######################
        # Transform
        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputConnection(obj_in.GetOutputPort())
        trans.SetTransform(T)
        trans.Update()

        mappers = vtk.vtkPolyDataMapper()
        mappers.SetInputData(trans.GetOutput())

        # actorText = vtk.vtkActor()
        # actorText.SetMapper(mappers)
        # actorText.SetTexture(texture)
        # actorText.GetProperty().SetColor(1, 1, 1)
        # actorText.GetProperty().SetAmbient(1.0)
        # actorText.GetProperty().SetSpecular(0)
        # actorText.GetProperty().SetDiffuse(0)

        actorGeom = vtk.vtkActor()
        actorGeom.SetMapper(mappers)
        # actorGeom.GetProperty().SetColor(1,1,1)
        # actorGeom.GetProperty().SetAmbient(1.0)
        # actorGeom.GetProperty().SetSpecular(0)
        # actorGeom.GetProperty().SetDiffuse(0)

        # ren.AddActor(actorText)
        ren.AddActor(actorGeom)

        # Output file names:
        oname_geometry = 'd:/data/temp/testgeometry' + str(idx) + ".png"
        #        oname_depth = output_base + subject_name + "/" + faces + "_depth" + str(idx) + ".png"
        #        oname_image = output_base + subject_name + "/" + faces + "_image" + str(idx) + ".png"
        #        oname_LM = output_base + subject_name + "/" + faces + "_LMtrans" + str(idx) + ".txt"
        #        oname_transform = output_base + subject_name + "/" + faces + "_Transform" + str(idx) + ".txt"

        # Create random rtransform
        rx, ry, rz, s, tx, ty = CreateRandomTransform()
        # rx,ry,rz,s,tx,ty = NoTransform()
        #    rx = -78
        #    ry = 34
        #    rz = -4

        T.Identity()
        T.RotateY(ry)
        T.RotateX(rx)
        T.RotateZ(rz)
        T.Update()

        trans.Update()
        #       trans2.Update()

        ## Something with camera...
        #    zmin = -150
        #    zmax = 150
        xmin = -150
        xmax = 150
        ymin = -150
        ymax = 150
        zmin = trans.GetOutput().GetBounds()[4]
        zmax = trans.GetOutput().GetBounds()[5]
        #        xmin = trans.GetOutput().GetBounds()[0]
        #        xmax= trans.GetOutput().GetBounds()[1]
        #        ymin = trans.GetOutput().GetBounds()[2]
        #        ymax= trans.GetOutput().GetBounds()[3]
        xlen = xmax - xmin
        ylen = ymax - ymin

        # trans.Update()

        cx = 0
        cy = 0
        extendFactor = 1.0
        slength = max([xlen, ylen]) * extendFactor
        zoomfac = winsize / slength

        ren.GetActiveCamera().SetParallelScale(slength / 2)
        ren.GetActiveCamera().SetPosition(cx, cy, 500)
        ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
        ren.GetActiveCamera().SetClippingRange(500 - zmax - slack, 500 - zmin + slack)

        ## Save textured image
        #      actorText.SetVisibility(True)
        # actorGeom.SetVisibility(false)
        # renWin.Render()
        # renWin.SetSize(winsize, winsize)
        # renWin.Render()

        # w2if = vtk.vtkWindowToImageFilter()
        # w2if.SetInput(renWin)
        # w2if.SetInputBufferTypeToRGB()

        # writer = vtk.vtkPNGWriter()
        # writer.SetInputConnection(w2if.GetOutputPort())
        # writer.SetFileName(oname_image)
        # writer.Write()

        ## Save depth
        # w2if.SetInputBufferTypeToZBuffer()
        # w2if.Update()

        # scale = vtk.vtkImageShiftScale()
        # scale.SetOutputScalarTypeToUnsignedChar()
        # scale.SetInputConnection(w2if.GetOutputPort())
        # scale.SetShift(0)
        # scale.SetScale(-255)

        # writer.SetInputConnection(scale.GetOutputPort())
        # writer.SetFileName(oname_depth)
        # writer.Write()
        # del scale

        ## Save geometry
        actorGeom.SetVisibility(True)
        # actorText.SetVisibility(False)
        renWin.Render()

        if (write_image_files):
            w2ifB = vtk.vtkWindowToImageFilter()
            w2ifB.SetInput(renWin)
            writerB = vtk.vtkPNGWriter()
            writerB.SetInputConnection(w2ifB.GetOutputPort())
            writerB.SetFileName(oname_geometry)
            writerB.Write()
            del writerB, w2ifB

        del trans, mappers, actorGeom, ren, renWin, T


#        del w2if, writer, w2ifB, writerB, trans, mappers, actorText, actorGeom, ren, renWin, T

def CreateRenderings1Subject2(config, subject_name, winsize, nLines):
    slack = 5
    write_image_files = True
    off_screen_rendering = True

    n_channels = 1  # for geometry rendering
    image_stack = np.zeros((nLines, winsize, winsize, n_channels), dtype=np.float32)

    obj_in = vtk.vtkOBJReader()
    obj_in.SetFileName(subject_name)
    obj_in.Update()

    # pd = vrmlin.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
    # pd.GetPointData().SetScalars(None)

    # Load texture
    # textureImage = vtk.vtkBMPReader()
    # textureImage.SetFileName(bmp_path)
    # textureImage.Update()

    # texture = vtk.vtkTexture()
    # texture.SetInterpolate(1)
    # texture.SetQualityTo32Bit()
    # texture.SetInputConnection(textureImage.GetOutputPort())

    # del textureImage

    ############## DO THINGS:    #################
    # Clean data
    # clean = vtk.vtkCleanPolyData()
    # clean.SetInputData(pd)
    # clean.Update()

    # Initialize Camera
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.GetActiveCamera().SetPosition(0, 0, 1)
    ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 1, 0)
    ren.GetActiveCamera().SetParallelProjection(1)

    # Initialize RenderWindow
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(winsize, winsize)
    renWin.SetOffScreenRendering(off_screen_rendering)

    # Initialize Transform
    T = vtk.vtkTransform()
    T.Identity()
    T.Update()

    ############## Assuming one mesh ######################
    # Transform
    trans = vtk.vtkTransformPolyDataFilter()
    trans.SetInputConnection(obj_in.GetOutputPort())
    trans.SetTransform(T)
    trans.Update()

    mappers = vtk.vtkPolyDataMapper()
    mappers.SetInputData(trans.GetOutput())

    # actorText = vtk.vtkActor()
    # actorText.SetMapper(mappers)
    # actorText.SetTexture(texture)
    # actorText.GetProperty().SetColor(1, 1, 1)
    # actorText.GetProperty().SetAmbient(1.0)
    # actorText.GetProperty().SetSpecular(0)
    # actorText.GetProperty().SetDiffuse(0)

    actorGeom = vtk.vtkActor()
    actorGeom.SetMapper(mappers)
    # actorGeom.GetProperty().SetColor(1,1,1)
    # actorGeom.GetProperty().SetAmbient(1.0)
    # actorGeom.GetProperty().SetSpecular(0)
    # actorGeom.GetProperty().SetDiffuse(0)

    # ren.AddActor(actorText)
    ren.AddActor(actorGeom)

    w2ifB = vtk.vtkWindowToImageFilter()
    w2ifB.SetInput(renWin)
    writerB = vtk.vtkPNGWriter()
    writerB.SetInputConnection(w2ifB.GetOutputPort())

    start = time.time()
    for idx in tqdm(range(0, nLines)):
        # print(idx)

        # Output file names:
        oname_geometry = config.temp_dir / ('rendering' + str(idx) + '.png')
        #        oname_depth = output_base + subject_name + "/" + faces + "_depth" + str(idx) + ".png"
        #        oname_image = output_base + subject_name + "/" + faces + "_image" + str(idx) + ".png"
        #        oname_LM = output_base + subject_name + "/" + faces + "_LMtrans" + str(idx) + ".txt"
        oname_transform = config.temp_dir / ('transform' + str(idx) + '.txt')

        # Create random rtransform
        rx, ry, rz, s, tx, ty = CreateRandomTransform()
        # rx,ry,rz,s,tx,ty = NoTransform()
        #    rx = -78
        #    ry = 34
        #    rz = -4

        T.Identity()
        T.RotateY(ry)
        T.RotateX(rx)
        T.RotateZ(rz)
        T.Update()

        trans.Update()
        #       trans2.Update()

        ## Something with camera...
        #    zmin = -150
        #    zmax = 150
        xmin = -150
        xmax = 150
        ymin = -150
        ymax = 150
        zmin = trans.GetOutput().GetBounds()[4]
        zmax = trans.GetOutput().GetBounds()[5]
        #        xmin = trans.GetOutput().GetBounds()[0]
        #        xmax= trans.GetOutput().GetBounds()[1]
        #        ymin = trans.GetOutput().GetBounds()[2]
        #        ymax= trans.GetOutput().GetBounds()[3]
        xlen = xmax - xmin
        ylen = ymax - ymin

        # trans.Update()

        cx = 0
        cy = 0
        extendFactor = 1.0
        slength = max([xlen, ylen]) * extendFactor
        zoomfac = winsize / slength

        ren.GetActiveCamera().SetParallelScale(slength / 2)
        ren.GetActiveCamera().SetPosition(cx, cy, 500)
        ren.GetActiveCamera().SetFocalPoint(cx, cy, 0)
        ren.GetActiveCamera().SetClippingRange(500 - zmax - slack, 500 - zmin + slack)

        ## Save textured image
        #      actorText.SetVisibility(True)
        # actorGeom.SetVisibility(false)
        # renWin.Render()
        # renWin.SetSize(winsize, winsize)
        # renWin.Render()

        # w2if = vtk.vtkWindowToImageFilter()
        # w2if.SetInput(renWin)
        # w2if.SetInputBufferTypeToRGB()

        # writer = vtk.vtkPNGWriter()
        # writer.SetInputConnection(w2if.GetOutputPort())
        # writer.SetFileName(oname_image)
        # writer.Write()

        ## Save depth
        # w2if.SetInputBufferTypeToZBuffer()
        # w2if.Update()

        # scale = vtk.vtkImageShiftScale()
        # scale.SetOutputScalarTypeToUnsignedChar()
        # scale.SetInputConnection(w2if.GetOutputPort())
        # scale.SetShift(0)
        # scale.SetScale(-255)

        # writer.SetInputConnection(scale.GetOutputPort())
        # writer.SetFileName(oname_depth)
        # writer.Write()
        # del scale

        ## Save geometry
        actorGeom.SetVisibility(True)
        # actorText.SetVisibility(False)
        renWin.Render()

        if write_image_files:
            w2ifB.Modified()  # Needed here else only first rendering is put to file
            writerB.SetFileName(str(oname_geometry))
            writerB.Write()
        else:
            w2ifB.Modified()  # Needed here else only first rendering is put to file
            w2ifB.Update()

        # add rendering to image stack
        im = w2ifB.GetOutput()
        rows, cols, _ = im.GetDimensions()
        sc = im.GetPointData().GetScalars()
        a = vtk_to_numpy(sc)
        components = sc.GetNumberOfComponents()
        a = a.reshape(rows, cols, components)
        a = np.flipud(a)

        # For now just take the first channel
        image_stack[idx, :, :, 0] = a[:, :, 0]

        # Save Transformation
        f = open(oname_transform, 'w')
        line = ' '.join(str(x) for x in np.array([rx,ry,rz,s,tx,ty]))
        f.write(line)
        f.close()

    end = time.time()
    print("Pure rendering generation time: " + str(end - start))

    del writerB, w2ifB
    del trans, mappers, actorGeom, ren, renWin, T
    #        del w2if, writer, w2ifB, writerB, trans, mappers, actorText, actorGeom, ren, renWin, T

    return image_stack


def test_obj_importer(subject_name):
    obj_in = vtk.vtkOBJImporter()
    obj_in.SetFileName(subject_name)
    obj_in.SetFileNameMTL('D:/Data/temp/MartinStandard.mtl')
    obj_in.SetTexturePath('D:/Data/temp/')
    obj_in.Update()

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    # ren.GetActiveCamera().SetPosition(0, 0, 1)
    # ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    # ren.GetActiveCamera().SetViewUp(0, 1, 0)
    # ren.GetActiveCamera().SetParallelProjection(1)

    winsize = 512

    # Initialize RenderWindow
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(winsize, winsize)

    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)
    obj_in.SetRenderWindow(renWin)
    obj_in.Update()

    renWin.Render()
    # iren.Start()

    write_image_files = True

    if write_image_files:
        w2ifB = vtk.vtkWindowToImageFilter()
        w2ifB.SetInput(renWin)
        writerB = vtk.vtkPNGWriter()
        writerB.SetInputConnection(w2ifB.GetOutputPort())
        writerB.SetFileName('D:/Data/temp/MartinStandard_imageWrite.png')
        writerB.Write()

        # test numpy code
        # dobj = vtk.vtkImageData()
        # dobj.DeepCopy(w2ifB.GetOutput())
        # ds1 = dsa.WrapDataObject(dobj)
        # elev_copy = np.copy(ds1.PointData['Elevation'])
        # elev_copy[1] = np.nan

        im = w2ifB.GetOutput()
        rows, cols, _ = im.GetDimensions()
        sc = im.GetPointData().GetScalars()
        a = vtk_to_numpy(sc)
        components = sc.GetNumberOfComponents()
        a = a.reshape(rows, cols, components)
        a = np.flipud(a)
        # assert a.shape == im.GetDimensions()

        plt.figure()
        plt.imshow(a)
        plt.axis('off')
        plt.ioff()
        plt.show()

        del writerB, w2ifB


def test_create_rendering(subject_name):
    winsize = 256
    nLines = 100

    print('Start rendering')
    start = time.time()
    image_stack = CreateRenderings1Subject2(subject_name, winsize, nLines)
    end = time.time()
    print("Rendering Generation time: " + str(end - start))

    im = np.zeros((winsize, winsize, 3), dtype=np.float32)
    # Visualise rendering number 17
    im[:, :, 0] = image_stack[17, :, :, 0] / 255  # hacky way to convert 1 channel image to RGB
    im[:, :, 1] = image_stack[17, :, :, 0] / 255
    im[:, :, 2] = image_stack[17, :, :, 0] / 255
    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.ioff()
    plt.show()


# -------------------------------------------------------------
# Find heatmap maxima
# -------------------------------------------------------------
def gaussian(center_x, center_y):
    """Returns a Gaussian function with the given parameters"""
    width_x = 3  # float(width)
    width_y = 3  # float(width)
    return lambda x, y: 1 * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)


def moments(data):
    """Returns (x, y, width)
    the Gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    xx, yy = np.indices(data.shape)
    x = (xx * data).sum() / total
    y = (yy * data).sum() / total
    return x, y


def fit_gaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    error_function = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                        data)
    p, success = optimize.leastsq(error_function, params)
    return p


def find_heat_map_maxima(heatmaps, sigma=None, method="simple"):
    """ heatmaps: (#LM, hm_size,hm_size) """
    out_dim = heatmaps.shape[0]  # number of landmarks
    hm_size = heatmaps.shape[1]
    # coordinates = np.zeros((out_dim, 2), dtype=np.float32)
    coordinates = np.zeros((out_dim, 3), dtype=np.float32)

    # TODO this entire method needs cleaning a better 1. order momentum maximum finder
    # simple: Use only maximum pixel value in HM
    if method == "simple":
        for k in range(out_dim):
            hm = copy.copy(heatmaps[k, :, :])
            highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
            px = highest_idx[0]
            py = highest_idx[1]
            value = hm[px, py]
            # print(highest_idx, ' ', value)
            # if np.argmax(hm) > 0.5:
            coordinates[k, :] = (px, py, value)

            #if value > 0.5:
            #    coordinates[k, :] = highest_idx
            #else:
            #    coordinates[k, :] = (np.nan, np.nan)

    elif method == "FitGauss":
        for k in range(out_dim):
            hm = copy.copy(heatmaps[k, :, :])
            (x, y) = fit_gaussian(hm)

            if np.argmax(hm) > 0.5:
                coordinates[k, :] = (x, y)
            else:
                coordinates[k, :] = (np.nan, np.nan)

    return coordinates


def generate_image_with_heatmap_maxima(image, heat_map):
    im_size = image.shape[0]
    hm_size = heat_map.shape[2]
    i = image.copy()

    coordinates = find_heat_map_maxima(heat_map, method='simple')
    # coordinates = find_heat_map_maxima(heat_map, method='FitGauss')

    # the predicted heat map is sometimes smaller than the input image
    factor = im_size / hm_size
    for c in range(coordinates.shape[0]):
        px = coordinates[c][0]
        py = coordinates[c][1]
        if not np.isnan(px) and not np.isnan(py):
            cx = int(px * factor)
            cy = int(py * factor)
            for x in range(cx-2, cx+2):
                for y in range(cy-2, cy+2):
                    i[x, y, 0] = 0
                    i[x, y, 1] = 0
                    i[x, y, 2] = 1  # blue
    return i


def show_image_and_heatmap(image, heat_map):
    # heat_map = heat_map.cpu()
    heat_map = heat_map.numpy()
    im_size = image.size(2)
    hm_size = heat_map.shape[2]

    # Super hacky way to convert gray to RGB
    # show first image in batch
    i = np.zeros((im_size, im_size, 3))
    i[:, :, 0] = image[0, :, :]
    i[:, :, 1] = image[0, :, :]
    i[:, :, 2] = image[0, :, :]

    # Generate combined heatmap image in RGB channels.
    # This must be possible to do smarter - Alas! My Python skillz are lacking
    hm = np.zeros((hm_size, hm_size, 3))
    n_lm = heat_map.shape[0]
    for lm in range(n_lm):
        r = random.random()  # generate random colour placed on the unit sphere in RGB space
        g = random.random()
        b = random.random()
        l = math.sqrt(r * r + g * g + b * b)
        r = r / l;
        g = g / l;
        b = b / l;
        hm[:, :, 0] = hm[:, :, 0] + heat_map[lm, :, :] * r
        hm[:, :, 1] = hm[:, :, 1] + heat_map[lm, :, :] * g
        hm[:, :, 2] = hm[:, :, 2] + heat_map[lm, :, :] * b

    im_marked = generate_image_with_heatmap_maxima(i, heat_map)

    plt.figure()
    plt.imshow(i)
    plt.figure()
    plt.imshow(hm)
    plt.figure()
    plt.imshow(im_marked)
    plt.axis('off')
    plt.ioff()
    plt.show()


def find_maxima_in_batch_of_heatmaps(heatmaps, cur_id, heatmap_maxima):
    write_heatmaps = False;
    heatmaps = heatmaps.numpy()
    batch_size = heatmaps.shape[0]

    for idx in range(batch_size):
        if write_heatmaps:
            name_hm_maxima = config.temp_dir / ('hm_maxima' + str(cur_id + idx) + '.txt')
            f = open(name_hm_maxima, 'w')

        coordinates = find_heat_map_maxima(heatmaps[idx, :, :, :], method='simple')
        for lm_no in range(coordinates.shape[0]):
            px = coordinates[lm_no][0]
            py = coordinates[lm_no][1]
            value = coordinates[lm_no][2]
            heatmap_maxima[lm_no, cur_id+idx, :] = (px, py, value)
            if write_heatmaps:
                out_str = str(px) + ' ' + str(py) + ' ' + str(value) + '\n'
                f.write(out_str)

        if write_heatmaps:
            f.close()


def prediction(config, device, model, image_stack):
    n_views = config['data_loader']['args']['n_views']
    batch_size = config['data_loader']['args']['batch_size']
    n_landmarks = config['arch']['args']['n_landmarks']

    show_result_image = False
    heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

    # process the views in batch sized chunks
    cur_id = 0
    while cur_id + batch_size <= n_views:
        cur_images = image_stack[cur_id:cur_id + batch_size, :, :, :]

        data = torch.from_numpy(cur_images)
        # data = torch.from_numpy(image_stack)
        data = data.permute(0, 3, 1, 2)  # from NHWC to NCHW

        data = data / 255

        with torch.no_grad():
            print('predicting heatmaps for batch ', cur_id, ' to ', cur_id + batch_size)
            start = time.time()
            data = data.to(device)
            output = model(data)
            end = time.time()
            print("Model prediction time: " + str(end - start))

            if cur_id == 0 and show_result_image:
                image = data[0, :, :, :].cpu()
                heat_map = output[1, 0, :, :, :].cpu()
                show_image_and_heatmap(image, heat_map)

            # output [stack (0 or 1), batch, lm, hm_size, hm_size]
            heatmaps = output[1, :, :, :, :].cpu()
            find_maxima_in_batch_of_heatmaps(heatmaps, cur_id, heatmap_maxima)

        cur_id = cur_id + batch_size

    return heatmap_maxima

def predict_one_subject(config):
    #subject_name = 'I:/Data/temp/MartinStandard.obj'
    subject_name = 'D:/Data/temp/MartinStandard.obj'
    # subject_name = 'D:/Data/temp/20130218092022_standard.obj'
    # subject_name = 'D:/Data/temp/20140508111926_standard.obj'

    logger = config.get_logger('test')
    # build model architecture, then print to console

    print('Initialising model')
    model = config.initialize('arch', module_arch)
    logger.info(model)

    print('Loading checkpoint')
    image_channels = n_views = config['data_loader']['args']['image_channels']
    if image_channels == "geometry":
        check_point_name = 'saved/trained/MVLMModel_DTU3D_geometry.pth'
    elif image_channels == "RGB":
        check_point_name = 'saved/trained/MVLMModel_DTU3D_RGB.pth'

    logger.info('Loading checkpoint: {} ...'.format(check_point_name))

    if config['n_gpu'] < 1:
        checkpoint = torch.load(check_point_name, map_location=torch.device('cpu'))  # force the model to the cpu
    else:
        checkpoint = torch.load(check_point_name)

    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    if config['n_gpu'] < 1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    render_3d = Render3D(config)
    image_stack, transform_stack = render_3d.render_3d_file(subject_name)

    heatmap_maxima = prediction(config, device, model, image_stack)

    u3d = Utils3D(config)
    u3d.heatmap_maxima = heatmap_maxima
    #u3d.read_heatmap_maxima()
    # u3d.read_3d_transformations()
    u3d.transformations_3d = transform_stack
    u3d.compute_lines_from_heatmap_maxima()
    #u3d.visualise_one_landmark_lines(40, 'saved/temp/DeepMVLM_DTU3D/0904_104414')
    u3d.compute_all_landmarks_from_view_lines()
    u3d.project_landmarks_to_surface(subject_name)
    u3d.write_landmarks_as_vtk_points()


def test_utils_3d(config):
    subject_name = 'D:/Data/temp/MartinStandard.obj'
    u3d = Utils3D(config)
    u3d.read_heatmap_maxima('saved/temp/DeepMVLM_DTU3D/0904_104414')  # force to read from saved data
    # u3d.read_heatmap_maxima()
    u3d.read_3d_transformations('saved/temp/DeepMVLM_DTU3D/0904_104414')  # force to read from saved data
    u3d.compute_lines_from_heatmap_maxima()
    u3d.visualise_one_landmark_lines(40, 'saved/temp/DeepMVLM_DTU3D/0904_104414')
    u3d.compute_all_landmarks_from_view_lines()
    u3d.project_landmarks_to_surface(subject_name)
    u3d.write_landmarks_as_vtk_points('saved/temp/DeepMVLM_DTU3D/0904_104414')


def test_render_3d(config):
    subject_name = 'D:/Data/temp/MartinStandard.obj'
    render_3d = Render3D(config)
    image_stack, transformation_stack = render_3d.render_3d_file(subject_name)

def test_moment_based_max():
    sz = 3
    a_len = 2 * sz + 1
    w = np.arange(a_len)
    a = np.array([0, 1, 2, 1, 0, 0, 0])
    s = np.sum(np.multiply(w, a))
    ss = np.sum(a)
    pos = s/ss - sz
    print(pos)

def make_gaussian(self, height, width, sigma=3, center=None):
    """
    Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def test_heatmap_max_finding():
    hm_size = 256
    max_length = hm_size
    x = 117
    y = 76
    hm = np.zeros((hm_size, hm_size), dtype=np.float32)
    s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
    hm[:, :] = make_gaussian(hm_size, hm_size, sigma=s, center=(x, y))

    highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
    px = highest_idx[0]
    py = highest_idx[1]
    print(px, py)

    sz = 5
    a_len = 2 * sz + 1
    if px > sz and hm_size - px > sz and py > sz and hm_size - py > sz:
        slc = hm[px - sz:px + sz + 1, py - sz:py + sz + 1]
        ar = np.arange(a_len)
        sum_x = np.sum(slc, axis=0)
        s = np.sum(np.multiply(ar, sum_x))
        ss = np.sum(sum_x)
        pos = s / ss - sz
        px = px + pos

        sum_y = np.sum(slc, axis=1)
        s = np.sum(np.multiply(ar, sum_y))
        ss = np.sum(sum_y)
        pos = s / ss - sz
        py = py + pos
    print(px, py)


def main():
    subject_name = 'D:/Data/temp/MartinStandard.obj'
    # test_obj_importer(subject_name)
    test_create_rendering(subject_name)


if __name__ == '__main__':
    # main()

    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    # predict_one_subject(config)
    # test_utils_3d(config)
    # test_render_3d(config)
    # test_moment_based_max()
    test_heatmap_max_finding()