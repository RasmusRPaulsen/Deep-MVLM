import numpy as np
import vtk
import os


class Utils3D:
    def __init__(self, config):
        self.config = config
        self.heatmap_maxima = None
        self.transformations_3d = None
        self.lm_start = None
        self.lm_end = None
        self.landmarks = None

    def read_heatmap_maxima(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_landmarks = self.config['arch']['args']['n_landmarks']
        n_views = self.config['data_loader']['args']['n_views']

        # [n_landmarks, n_views, x, y, value]
        self.heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/hm_maxima' + str(idx) + '.txt'
            with open(name_hm_maxima) as f:
                id_lm = 0
                for line in f:
                    line = line.strip("/n")
                    x, y, val = np.double(line.split(" "))
                    self.heatmap_maxima[id_lm, idx, :] = (x, y, val)
                    id_lm = id_lm + 1
                    if id_lm > n_landmarks:
                        print('Too many landmarks in file ', name_hm_maxima)
                        break

            if id_lm != n_landmarks:
                print('Too few landmarks in file ', name_hm_maxima)

    def read_3d_transformations(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_views = self.config['data_loader']['args']['n_views']

        # [n_views, rx, ry, rz, s, tx, ty]
        self.transformations_3d = np.zeros((n_views, 6))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/transform' + str(idx) + '.txt'
            rx, ry, rz, s, tx, ty = np.loadtxt(name_hm_maxima)
            self.transformations_3d[idx, :] = (rx, ry, rz, s, tx, ty)

    # Each maxima in a heatmap corresponds to a line in 3D space of the original 3D shape
    # This function transforms the maxima to (start point, end point) pairs
    def compute_lines_from_heatmap_maxima(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        n_views = self.heatmap_maxima.shape[1]

        self.lm_start = np.zeros((n_landmarks, n_views, 3))
        self.lm_end = np.zeros((n_landmarks, n_views, 3))

        img_size = self.config['data_loader']['args']['image_size']
        hm_size = self.config['data_loader']['args']['heatmap_size']
        winsize = img_size

        # TODO these fixed values should probably be in a config file
        x_min = -150
        x_max = 150
        y_min = -150
        y_max = 150
        x_len = x_max - x_min
        y_len = y_max - y_min

        pd = vtk.vtkPolyData()
        for idx in range(n_views):
            rx, ry, rz, s, tx, ty = self.transformations_3d[idx, :]

            # Set transformation matrix in vtk
            t = vtk.vtkTransform()
            t.Identity()
            t.Update()

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()

            for lm_no in range(n_landmarks):
                # [n_landmarks, n_views, x, y, value]
                y = self.heatmap_maxima[lm_no, idx, 0]
                x = self.heatmap_maxima[lm_no, idx, 1]
                # value = self.heatmap_maxima[lm_no, idx, 2]

                #  Extract just one landmark and scale it according to heatmap and image sizes
                y = y / hm_size * img_size
                x = x / hm_size * img_size

                # Making end points of line in world coordinates
                p_wc_s = np.zeros((3, 1))
                p_wc_e = np.zeros((3, 1))

                p_wc_s[0] = (x / winsize) * x_len + x_min
                p_wc_s[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_s[2] = 500

                p_wc_e[0] = (x / winsize) * x_len + x_min
                p_wc_e[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_e[2] = -500

                # Insert line into vtk-framework to transform
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

                lines.InsertNextCell(2)
                pid = points.InsertNextPoint(p_wc_s)
                lines.InsertCellPoint(pid)
                pid = points.InsertNextPoint(p_wc_e)
                lines.InsertCellPoint(pid)

                pd.SetPoints(points)
                del points
                pd.SetLines(lines)
                del lines

                # Do inverse transform into original space
                tfilt = vtk.vtkTransformPolyDataFilter()
                tfilt.SetTransform(t.GetInverse())
                tfilt.SetInputData(pd)
                tfilt.Update()

                lm_out = vtk.vtkPolyData()
                lm_out.DeepCopy(tfilt.GetOutput())

                self.lm_start[lm_no, idx, :] = lm_out.GetPoint(0)
                self.lm_end[lm_no, idx, :] = lm_out.GetPoint(1)

                del tfilt
            del t
        del pd

    def visualise_one_landmark_lines(self, lm_no, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Writing to', dir_name)

        lm_name = dir_name + '/lm_lines_' + str(lm_no) + '.vtk'

        n_views = self.heatmap_maxima.shape[1]
        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        verts = vtk.vtkCellArray()
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfValues(2 * n_views)

        for idx in range(n_views):
            lines.InsertNextCell(2)
            pid = points.InsertNextPoint(self.lm_start[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
            pid = points.InsertNextPoint(self.lm_end[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            scalars.SetValue(idx * 2, self.heatmap_maxima[lm_no, idx, 2])  # Color code according to maxima value
            scalars.SetValue(idx * 2 + 1, self.heatmap_maxima[lm_no, idx, 2])

        pd.SetPoints(points)
        del points
        pd.SetLines(lines)
        del lines
        pd.SetVerts(verts)
        del verts
        pd.GetPointData().SetScalars(scalars)
        del scalars

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(lm_name)
        writer.Write()

        del writer
        del pd

    # FROM: https://se.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space?focused
    # =5235003&tab=function"
    """
    Find intersection point of lines in 3D space, in the least squares sense.
    pa :          Nx3-matrix containing starting point of N lines
    pa :          Nx3-matrix containing end point of N lines
    p_intersect : Best intersection point of the N lines, in least squares sense.
    distances   : Distances from intersection point to the input lines
    Anders Eikenes, 2012 """
    def compute_intersection_between_lines(self, pa, pb):
        n_lines = pa.shape[0]
        si = pb - pa  # N lines described as vectors
        ni = np.divide(si, np.transpose(np.sqrt(np.sum(si ** 2, 1)) * np.ones((3, n_lines))))  # Normalize vectors
        nx = ni[:, 0]
        ny = ni[:, 1]
        nz = ni[:, 2]
        sxx = np.sum(nx ** 2 - 1)
        syy = np.sum(ny ** 2 - 1)
        szz = np.sum(nz ** 2 - 1)
        sxy = np.sum(np.multiply(nx, ny))
        sxz = np.sum(np.multiply(nx, nz))
        syz = np.sum(np.multiply(ny, nz))
        s = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
        cx = np.sum(np.multiply(pa[:, 0], (nx ** 2 - 1)) + np.multiply(pa[:, 1], np.multiply(nx, ny)) +
                    np.multiply(pa[:, 2], np.multiply(nx, nz)))
        cy = np.sum(np.multiply(pa[:, 0], np.multiply(nx, ny)) + np.multiply(pa[:, 1], (ny ** 2 - 1)) +
                    np.multiply(pa[:, 2], np.multiply(ny, nz)))
        cz = np.sum(np.multiply(pa[:, 0], np.multiply(nx, nz)) + np.multiply(pa[:, 1], np.multiply(ny, nz)) +
                    np.multiply(pa[:, 2], (nz ** 2 - 1)))

        c = np.array([[cx], [cy], [cz]])
        p_intersect = np.matmul(np.linalg.pinv(s), c)
        return p_intersect[:, 0]

    def compute_intersection_between_lines_ransac(self, pa, pb):
        # TODO parameters in config
        iterations = 100
        best_error = 100000000  # TODO should find a better initialiser
        best_p = (0, 0, 0)
        dist_thres = 10 * 10  # TODO should find a better way to esimtate dist_thres
        # d = 10  #
        n_lines = len(pa)
        d = n_lines / 3
        used_lines = -1

        for i in range(iterations):
            # get 3 random lines
            ran_lines = np.random.choice(range(n_lines), 3, replace=False)
            # Compute first estimate of intersection
            p_est = self.compute_intersection_between_lines(pa[ran_lines, :], pb[ran_lines, :])
            # Compute distance from all lines to intersection
            top = np.cross((np.transpose(p_est) - pa), (np.transpose(p_est) - pb))
            bottom = pb - pa
            distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2
            # number of inliners
            n_inliners = np.sum(distances < dist_thres)
            if n_inliners > d:
                # reestimate based on inliners
                idx = distances < dist_thres
                p_est = self.compute_intersection_between_lines(pa[idx, :], pb[idx, :])

                # Compute distance from all inliners to intersection
                top = np.cross((np.transpose(p_est) - pa[idx, :]), (np.transpose(p_est) - pb[idx, :]))
                bottom = pb[idx, :] - pa[idx, :]
                distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2

                # sum_squared = np.sum(np.square(distances)) / n_inliners
                sum_squared = np.sum(distances) / n_inliners
                if sum_squared < best_error:
                    best_error = sum_squared
                    best_p = p_est
                    used_lines = n_inliners

        if used_lines == -1:
            print('Ransac failed - estimating from all lines')
            best_p = self.compute_intersection_between_lines(pa, pb)
        # else:
        # print('Ransac error ', best_error, ' with ', used_lines, ' lines')

        return best_p, best_error

    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_quantiles(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        q = self.config['process_3d']['heatmap_max_quantile']
        threshold = np.quantile(max_values, q)
        idx = max_values > threshold
        # print('Using ', threshold, ' as threshold in heatmap maxima')
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new

    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_absolute_value(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        threshold = self.config['process_3d']['heatmap_abs_threshold']
        idx = max_values > threshold
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new

    # Each landmark can be computed by the intersection of the view lines going trough (or near) it
    def compute_all_landmarks_from_view_lines(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        self.landmarks = np.zeros((n_landmarks, 3))

        sum_error = 0
        for lm_no in range(n_landmarks):
            pa = self.lm_start[lm_no, :, :]
            pb = self.lm_end[lm_no, :, :]
            if self.config['process_3d']['filter_view_lines'] == "abs_value":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_absolute_value(lm_no, pa, pb)
            elif self.config['process_3d']['filter_view_lines'] == "quantile":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_quantiles(lm_no, pa, pb)
            p_intersect = (0, 0, 0)
            if len(pa) < 3:
                print('Not enough valid view lines for landmark ', lm_no)
            else:
                # p_intersect = self.compute_intersection_between_lines(pa, pb)
                p_intersect, best_error = self.compute_intersection_between_lines_ransac(pa, pb)
                sum_error = sum_error + best_error
            self.landmarks[lm_no, :] = p_intersect
        print("Ransac average error ", sum_error/n_landmarks)

    def multi_read_surface(self, file_name):
        clean_name, file_extension = os.path.splitext(file_name)
        if file_extension == ".obj":
            obj_in = vtk.vtkOBJReader()
            obj_in.SetFileName(file_name)
            obj_in.Update()
            pd = obj_in.GetOutput()
            return pd
        elif file_extension == ".wrl":
            vrmlin = vtk.vtkVRMLImporter()
            vrmlin.SetFileName(file_name)
            vrmlin.Update()
            pd = vrmlin.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
            return pd

    # Project found landmarks to closest point on the target surface
    def project_landmarks_to_surface(self, mesh_name):
        # obj_in = vtk.vtkOBJReader()
        # obj_in.SetFileName(mesh_name)
        # obj_in.Update()
        pd = self.multi_read_surface(mesh_name)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        # clean.SetInputConnection(pd.GetOutputPort())
        clean.Update()

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(clean.GetOutput())
        locator.SetNumberOfCellsPerBucket(1)
        locator.BuildLocator()

        projected_landmarks = np.copy(self.landmarks)
        n_landmarks = self.landmarks.shape[0]

        for i in range(n_landmarks):
            p = self.landmarks[i, :]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.reference(0)
            tcp = np.zeros(3)

            locator.FindClosestPoint(p, tcp, cell_id, sub_id, dist2)
            # print('Nearest point in distance ', np.sqrt(np.float(dist2)))
            projected_landmarks[i, :] = tcp

        self.landmarks = projected_landmarks
        del pd
        del clean
        del locator

    def write_landmarks_as_vtk_points(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Writing to', dir_name)

        lm_name = dir_name + '/lms_as_points.vtk'
        n_landmarks = self.heatmap_maxima.shape[0]

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        for lm_no in range(n_landmarks):
            pid = points.InsertNextPoint(self.landmarks[lm_no, :])
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)

        pd.SetPoints(points)
        del points
        pd.SetVerts(verts)
        del verts

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(lm_name)
        writer.Write()

        del writer
        del pd

    @staticmethod
    def write_landmarks_as_vtk_points_external(landmarks, file_name):
        n_landmarks = landmarks.shape[0]

        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()

        for lm_no in range(n_landmarks):
            pid = points.InsertNextPoint(landmarks[lm_no, :])
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)

        pd.SetPoints(points)
        del points
        pd.SetVerts(verts)
        del verts

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(file_name)
        writer.Write()

        del writer
        del pd

    @staticmethod
    def write_landmarks_as_text_external(landmarks, file_name):
        f = open(file_name, 'w')

        for lm in landmarks:
            px = lm[0]
            py = lm[0]
            pz = lm[0]
            out_str = str(px) + ' ' + str(py) + ' ' + str(pz) + '\n'
            f.write(out_str)

        f.close()
