import math as m
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import awkward as awk
import os
import uproot
from itertools import repeat
from multiprocessing import Pool, Value
import time

def drawLine(shape, img, tkdxy_err, max_dxy_err): #Draws a line whose magnitude is the magnitude of px and py
    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(200 * tkdxy_err/max_dxy_err)

    #Draws the line
    img.line(xy=shape, fill=grayscale, width=1)


def drawTriangle(lineshape, img, imagecenter, tkdxy_err, max_dxy_err, tk_pt, max_tk_pt): #Draw isoceles triangle where magnitude of the track is the height
    line_magnitude = m.sqrt((lineshape[1][0]-lineshape[0][0])**2 + (lineshape[1][1]-lineshape[0][1])**2)

    #Calculate magnitude of the base based on the pt of the track
    base_magnitude = (line_magnitude/6) * m.log(tkdxy_err)/m.log(max_dxy_err)

    #Calculate the angle of the rotated track
    tk_theta = calculateAngle(lineshape[1][0]-lineshape[0][0], lineshape[0][1]-lineshape[1][1])

    #Calculate corners of the triangle
    xcorner, ycorner = base_magnitude*m.sin(m.radians(tk_theta)), base_magnitude*m.cos(m.radians(tk_theta))
    initial_points = (lineshape[0][0], lineshape[0][1])

    if (90 < tk_theta and tk_theta <= 180) or (270 < tk_theta and tk_theta < 360):
        corner1 = (lineshape[1][0]-xcorner, lineshape[1][1]-ycorner)
        corner2 = (lineshape[1][0]+xcorner, lineshape[1][1]+ycorner)
    else:
        corner1 = (lineshape[1][0] - xcorner, lineshape[1][1] - ycorner)
        corner2 = (lineshape[1][0] + xcorner, lineshape[1][1] + ycorner)

    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(170 - (170 * m.log(tk_pt)/m.log(max_tk_pt)))

    #Draw triangle
    img.polygon(xy=[initial_points, corner1, corner2], fill=None, outline=grayscale)


def ObtainData(stree, btree, beamspot_coordinates, IsSignal=True):
    if str(type(stree)) == "<class 'awkward.highlevel.Array'>":
        if IsSignal == True:
            px = stree['vtx_tk_px']
            py = stree['vtx_tk_py']
            pt = stree['vtx_tk_pt']
            dxy = stree['vtx_tk_dxy']
            dxy_err = stree['vtx_tk_dxyerr']
            vtx_x = stree['vtx_x']
            vtx_y = stree['vtx_y']
        elif IsSignal == False:
            px = btree['vtx_tk_px']
            py = btree['vtx_tk_py']
            pt = btree['vtx_tk_pt']
            dxy = btree['vtx_tk_dxy']
            dxy_err = btree['vtx_tk_dxyerr']
            vtx_x = btree['vtx_x']
            vtx_y = btree['vtx_y']
        else:
            print("Error: Dataset must be True for signal or False for background")
            sys.exit()
    else:
        if IsSignal == True:
            px = stree['vtx_tk_px'].array()
            py = stree['vtx_tk_py'].array()
            pt = stree['vtx_tk_pt'].array()
            dxy = stree['vtx_tk_dxy'].array()
            dxy_err = stree['vtx_tk_dxyerr'].array()
            vtx_x = stree['vtx_x'].array()
            vtx_y = stree['vtx_y'].array()
        elif IsSignal == False:
            px = btree['vtx_tk_px'].array()
            py = btree['vtx_tk_py'].array()
            pt = btree['vtx_tk_pt'].array()
            dxy = btree['vtx_tk_dxy'].array()
            dxy_err = btree['vtx_tk_dxyerr'].array()
            vtx_x = btree['vtx_x'].array()
            vtx_y = btree['vtx_y'].array()
        else:
            print("Error: Dataset must be True for signal or False for background")
            sys.exit()
    #Fix Vtx_x and Vtx_y to be relative to beamspot center
    vtx_x = np.subtract(vtx_x, beamspot_coordinates[0])
    vtx_y = np.subtract(vtx_y, beamspot_coordinates[1])

    return px, py, pt, dxy, dxy_err, vtx_x, vtx_y


def calculateAngle(x, y): #Returns the 360 degree angle for a given 2D vector
    try:
        theta = m.degrees(m.atan2(y, x))
        if theta < 0:
            theta = 360 - abs(theta)
    except ZeroDivisionError:
        if y>0:
            theta = 90.0
        else:
            theta = 270.0

    return theta


def GetMaxVar(var): #Returns the maximum value of a particular variable
    maxL = []
    for a in var:
        for b in a:
            maxL.append(max(b))

    return max(maxL)


def FixLogSign(result, log):
    result_fixed = None
    if log > 0:
        if result < 0:
            result_fixed = -1 * result
        else:
            result_fixed = result
    elif log < 0:
        if result > 0:
            result_fixed = -1 * result
        else:
            result_fixed = result

    return result_fixed


def calculateXYImage(px, py, maxpx, maxpy, tk2_dx, tk2_dy, imagecenter, Pscale_multiplicity_constant, DXscale_multiplicity_constant): #Calculate the endpoints of the track lines for the images in x y space
    #Initialize the lists
    x_initial_list = []
    y_initial_list = []
    x_final_list = []
    y_final_list = []

    #Create track magnitude constant
    cx, cy = Pscale_multiplicity_constant*imagecenter/m.log(maxpx), Pscale_multiplicity_constant*imagecenter/m.log(maxpy)
    cdx, cdy = DXscale_multiplicity_constant, DXscale_multiplicity_constant

    #Resize the px and py components of the track in accordance to the constant and the log of px/max(px)
    for i in range(len(px)):
        clogx = cx*m.log(abs(px[i]))
        clogy = cy*m.log(abs(py[i]))
        clogdx = cdx*m.log(abs(tk2_dx[i]))
        clogdy = cdy*m.log(abs(tk2_dy[i]))

        #Fix log signs
        clogx = FixLogSign(clogx, px[i])
        clogy = FixLogSign(clogy, py[i])
        clogdx = FixLogSign(clogdx, tk2_dx[i])
        clogdy = FixLogSign(clogdy, tk2_dy[i])

        x_initial_list.append(round(imagecenter+clogdx))
        y_initial_list.append(round(imagecenter-clogdy))
        x_final_list.append(round(x_initial_list[i]+clogx))
        y_final_list.append(round(y_initial_list[i]-clogy))

    return x_initial_list, y_initial_list, x_final_list, y_final_list


def calculateSecondaryDXY(tk_px, tk_py, tk_dxy, vtx_x, vtx_y): #Calculates the track dxy with respect to the secondary vertex
    #Initialize list and variables
    vtx_mag = m.sqrt(vtx_x**2 + vtx_y**2)
    vtx_theta = calculateAngle(vtx_x, vtx_y)
    tkp_theta = []
    tk2_dxy = []
    tk2_dx = []
    tk2_dy = []

    #Calculate the secondary dxy
    for i in range(len(tk_px)):
        tkp_theta.append(calculateAngle(tk_px[i], tk_py[i]))

        if (tkp_theta[i] >= 0 and tkp_theta[i] <= 90) or (tkp_theta[i] > 180 and tkp_theta[i] <= 270):
            psi = vtx_theta - tkp_theta[i] + 90
            omega = 90 - tkp_theta[i]
        else:
            psi = vtx_theta + tkp_theta[i] - 90
            omega = tkp_theta[i] - 90

        tk2_dxy.append(tk_dxy[i] - tk_dxy[i]*vtx_mag*m.cos(m.radians(psi)))
        tk2_dx.append(tk2_dxy[i]*m.cos(m.radians(omega)))
        tk2_dy.append(tk2_dxy[i]*m.sin(m.radians(omega)))

    return tk2_dx, tk2_dy


def RotateComponents(x_initial_list, y_initial_list, x_final_list, y_final_list, vtx_theta): #Rotates the tracks so that the up direction points away from the primary vertex
    #Initalize the lists
    x_final_list_rotated = []
    y_final_list_rotated = []
    x_final_list_rotated_fixed = []
    y_final_list_rotated_fixed = []

    #Rotate the x and y components
    for i in range(len(x_initial_list)):
        deltax = x_final_list[i]-x_initial_list[i]
        deltay = y_initial_list[i]-y_final_list[i]
        rotation_angle = 180 + vtx_theta
        x_final_list_rotated.append(deltax*m.cos(m.radians(rotation_angle)) - deltay*m.sin(m.radians(rotation_angle)))
        y_final_list_rotated.append(deltax * m.sin(m.radians(rotation_angle)) + deltay * m.cos(m.radians(rotation_angle)))
        x_final_list_rotated_fixed.append(x_initial_list[i] + x_final_list_rotated[i])
        y_final_list_rotated_fixed.append(y_initial_list[i] + y_final_list_rotated[i])

    return x_final_list_rotated_fixed, y_final_list_rotated_fixed


def createVertexImage(stree, btree, beamspot_coordinates, event, vertex, w=224, h=224, rootFileName=None, ImageFilePath=None,
                      IsSignal=True, showTitle=False, drawCenter=False, maxData=None, parallel=False):
    #Set up counter for creating N vertices
    if parallel:
        global counter
        global approx_time_coefficient
        global Np
        global start_time

    # Obtain momentum data
    px, py, pt, dxy, dxy_err, vtx_x, vtx_y = ObtainData(stree, btree, beamspot_coordinates, IsSignal)

    # Obtain max data
    if maxData is None:
        Smaxpx, Smaxpy, Smaxpt, Smax_dxy_err = GetMaxVar(stree['vtx_tk_px'].array()), GetMaxVar(stree['vtx_tk_py'].array()), \
                                               GetMaxVar(stree['vtx_tk_pt'].array()), GetMaxVar(stree['vtx_tk_dxyerr'].array())
        Bmaxpx, Bmaxpy, Bmaxpt, Bmax_dxy_err = GetMaxVar(btree['vtx_tk_px'].array()), GetMaxVar(btree['vtx_tk_py'].array()), \
                                               GetMaxVar(btree['vtx_tk_pt'].array()), GetMaxVar(btree['vtx_tk_dxyerr'].array())
        maxpx, maxpy, maxpt, max_dxy_err = max(Smaxpx, Bmaxpx), max(Smaxpy, Bmaxpy), max(Smaxpt, Bmaxpt), max(Smax_dxy_err,
                                                                                                              Bmax_dxy_err)
    else:
        maxpx, maxpy, maxpt, max_dxy_err = maxData['maxpx'], maxData['maxpy'], maxData['maxpt'], maxData['max_dxy_err']

    # Obtain vertex data
    try:
        px0 = px[event][vertex]
        py0 = py[event][vertex]
        pt0 = pt[event][vertex]
        dxy0 = dxy[event][vertex]
        vtx_x0 = vtx_x[event][vertex]
        vtx_y0 = vtx_y[event][vertex]
        dxy_err0 = dxy_err[event][vertex]
    except ValueError:
        if IsSignal == True:
            print("There is no data in the signal file for event number " + str(event))
            return
        else:
            print("There is no data in the background file for event number " + str(event))
            return

    # Calculate secondary dxy
    tk2_dx, tk2_dy = calculateSecondaryDXY(px0, py0, dxy0, vtx_x0, vtx_y0)

    # Create image
    center = int(w / 2)
    x_initial_list, y_initial_list, x_final_list, y_final_list = calculateXYImage(px0, py0, maxpx, maxpy, tk2_dx,
                                                                                  tk2_dy, center, 4 / 5, 1)
    img = Image.new(mode='L', size=(w, h), color=255)
    img1 = ImageDraw.Draw(img)

    if drawCenter:
        # Draw circle at image center to represent secondary vertex
        img1.ellipse((center - 2, center - 2, center + 2, center + 2), fill=None, outline=0)

    #Draw text in upper right hand corner to specify signal/background and also the event and vertex
    if showTitle == False:
        pass
    else:
        if IsSignal == True:
            title = "Signal \nEvent = {" + str(event) + "} \nVertex = {" + str(vertex) + "}"
        else:
            title = "Background \nEvent = {" + str(event) + "} \nVertex = {" + str(vertex) + "}"
        img1.text((w - 80, 0), title, align='center')

    # Rotate components so that up direction points away from primary vertex
    vtx_theta = calculateAngle(vtx_x0, vtx_y0)
    x_final_list_rotated, y_final_list_rotated = RotateComponents(x_initial_list, y_initial_list, x_final_list,
                                                                  y_final_list, vtx_theta)

    # Draw Triangles
    for i in range(len(x_final_list)):
        shape = [(x_initial_list[i], y_initial_list[i]), (x_final_list_rotated[i], y_final_list_rotated[i])]
        drawTriangle(shape, img1, center, dxy_err0[i], max_dxy_err, pt0[i], maxpt)

    if ImageFilePath != None:
        if IsSignal == True:
            if rootFileName != None:
                r1 = os.path.basename(os.path.normpath(rootFileName))
                r2 = r1.split('.')[0]
                filepath = os.path.join(ImageFilePath, '{}x{}'.format(w, h), 'signal', '{} Evt {} Vtx {}.png'.format(r2, event, vertex))
            else:
                filepath = os.path.join(ImageFilePath, '{}x{}'.format(w, h), 'signal', 'Evt {} Vtx {}.png'.format(event, vertex))
        else:
            if rootFileName != None:
                r1 = os.path.basename(os.path.normpath(rootFileName))
                r2 = r1.split('.')[0]
                filepath = os.path.join(ImageFilePath, '{}x{}'.format(w, h), 'background', '{} Evt {} Vtx {}.png'.format(r2, event, vertex))
            else:
                filepath = os.path.join(ImageFilePath, '{}x{}'.format(w, h), 'background', 'Evt {} Vtx {}.png'.format(event, vertex))
        img.save(filepath)
    else:
        img.show()

    #Update counter for parallel processes
    if parallel:
        with counter.get_lock():
            counter.value+=1
        if counter.value%49 == 0 or counter.value%50 == 0 or counter.value%51 == 0:
            elapsed_time = time.time() - start_time
            approx_time_remaining = round(elapsed_time / approx_time_coefficient - elapsed_time, 2)
            if IsSignal:
                print('{}/{} Signal Images have been created. Elapsed time for this ROOT: {}. Approximate time remaining for this ROOT: {}'.format(
                        counter.value, Np, elapsed_time, approx_time_remaining))
            else:
                print('{}/{} Background Images have been created. Elapsed time for this ROOT: {}. Approximate time remaining for this ROOT: {}'.format(
                        counter.value, Np, elapsed_time, approx_time_remaining))
            approx_time_coefficient += (50 / Np)


def Observe_N_Vertices(N, stree=None, btree=None, beamspot_coordinates=None, w=224, h=224, IsSignal=True, ImageFilePath=None,
                       showTitle=False, drawCenter=False, maxData=None, rootFileName=None, parallel=False, start=0): #View or save N signal or background vertices, set N='all' to view all vertices
    #Observe N signal vertices
    if IsSignal:
        if N == 'all':
            N=len(awk.flatten(stree['vtx_tk_px'].array()))
        if parallel: #Run function in parallel
            #Create approximate time coefficient for future time calculation
            approx_time_coefficient = 50 / N

            #Set up counter
            counter = Value('i', 0)

            #Set up parallel processing by obtaining cpu count and creating iterable arguments
            workers = os.cpu_count()
            event, vertex = ObtainEventToVertexIterables(stree['vtx_tk_px'].array(), N, start)
            args_iter = zip(repeat(stree), repeat(btree), repeat(beamspot_coordinates), event, vertex, repeat(w), repeat(h),
                            repeat(rootFileName), repeat(ImageFilePath), repeat(IsSignal), repeat(showTitle), repeat(drawCenter),
                            repeat(maxData), repeat(True))

            #Start timer and begin parallel processing
            start_time = time.time()
            with Pool(workers, initializer=init, initargs=(counter, N, approx_time_coefficient, start_time, )) as p:
                results = p.starmap_async(createVertexImage, iterable=args_iter, error_callback=ParallelErrorCallback)
                results.get()

        else: #Run function in standard
            n = 1
            start_time = time.time()
            approx_time_coefficient = 50 / N
            while n <= N:
                for event in range(len(stree['vtx_tk_px'].array())):
                    if n == N+1:
                        break
                    for vertex in range(len(stree['vtx_tk_px'].array()[event])):
                        createVertexImage(stree, btree, beamspot_coordinates, event, vertex, w=w, h=h, ImageFilePath=ImageFilePath,
                                          IsSignal=IsSignal, showTitle=showTitle, drawCenter=drawCenter,
                                          maxData=maxData, rootFileName=rootFileName)
                        n += 1

                if n%49 ==0 or n%50 == 0 or n%51 == 0:
                    elapsed_time = time.time() - start_time
                    approx_time_remaining = round(elapsed_time / approx_time_coefficient - elapsed_time, 2)
                    print(
                        '{}/{} Signal Images have been created. Elapsed time for this ROOT: {}. Approximate time remaining for this ROOT: {}'.format(
                            n, N, elapsed_time, approx_time_remaining))
                    approx_time_coefficient += (50 / N)


    #Observe N background vertices
    else:
        if N == 'all':
            N = len(awk.flatten(btree['vtx_tk_px'].array()))
        if parallel:  # Run function in parallel
            # Create approximate time coefficient for future time calculation
            approx_time_coefficient = 50 / N

            # Set up counter
            counter = Value('i', 0)

            # Set up parallel processing by obtaining cpu count and creating iterable arguments
            workers = os.cpu_count()
            event, vertex = ObtainEventToVertexIterables(btree['vtx_tk_px'].array(), N, start)
            args_iter = zip(repeat(stree), repeat(btree), repeat(beamspot_coordinates), event, vertex, repeat(w), repeat(h),
                            repeat(rootFileName), repeat(ImageFilePath), repeat(IsSignal), repeat(showTitle), repeat(drawCenter),
                            repeat(maxData), repeat(True))

            # Start timer and begin parallel processing
            start_time = time.time()
            with Pool(workers, initializer=init, initargs=(counter, N, approx_time_coefficient, start_time, )) as p:
                results = p.starmap_async(createVertexImage, iterable=args_iter, error_callback=ParallelErrorCallback)
                results.get()

        else:  # Run function in standard
            n = 1
            start_time = time.time()
            approx_time_coefficient = 50 / N
            while n <= N:
                for event in range(len(btree['vtx_tk_px'].array())):
                    if n == N+1:
                        break
                    for vertex in range(len(btree['vtx_tk_px'].array()[event])):
                        createVertexImage(stree, btree, beamspot_coordinates, event, vertex, w=w, h=h, ImageFilePath=ImageFilePath,
                                          IsSignal=IsSignal, showTitle=showTitle, drawCenter=drawCenter,
                                          maxData=maxData, rootFileName=rootFileName)
                        n += 1

                if n%49 ==0 or n%50 == 0 or n%51 == 0:
                    elapsed_time = time.time() - start_time
                    approx_time_remaining = round(elapsed_time / approx_time_coefficient - elapsed_time, 2)
                    print(
                        '{}/{} Signal Images have been created. Elapsed time for this ROOT: {}. Approximate time remaining for this ROOT: {}'.format(
                            n, N, elapsed_time, approx_time_remaining))
                    approx_time_coefficient += (50 / N)


def ParallelErrorCallback():
    return


def init(arg1, arg2, arg3, arg4): #Function used for global counter when running in parallel
    global counter
    global Np
    global approx_time_coefficient
    global start_time
    counter = arg1
    Np = arg2
    approx_time_coefficient = arg3
    start_time = arg4


def ObtainEventToVertexIterables(track_data, n_vertices='all', start=0): #Takes a root array {stree['vtx_tk_px'].array()} and creates two lists, one for the event number and one for the vertex number for all vertices
    #Instantiate lists
    vertices = []
    event_l = []

    #Fill lists with event/vertex number data
    if n_vertices == 'all':
        for event in range(len(track_data)):
            for vertex in range(len(track_data[event])):
                vertices.append(vertex)
    else:
        n=1
        for event in range(len(track_data)):
            while n <= n_vertices:
                for vertex in range(len(track_data[event])):
                    vertices.append(vertex)
                    n+=1

    event_number = -1
    for vertex in vertices:
        if vertex == 0:
            event_number += 1
            event_l.append(event_number)
        else:
            event_l.append(event_number)

    if start != 0: #Start the dataset at a particular vertex
        counter = 0
        for i in range(len(event_l)):
            if event_l[i] < start:
                counter += 1
            else:
                break

        del event_l[:counter]
        del vertices[:counter]

        if len(event_l) == 0:
            print("WARNING: The 'start' variable passed into ObtainEventToVertexIterables was too large. This resulted in an empty event list."
                " The script will now be killed and it is recommended that you reduce 'start'.")
            sys.exit()


    return event_l, vertices


def CountTrackQuadrants(stree, btree, beamspot_coordinates, w=224, IsSignal=True):
    #Initiate quadrant list
    quadrants = []
    if IsSignal:
        data = 'signal'
    else:
        data = 'background'

    # Obtain momentum data
    px, py, pt, dxy, dxy_err, vtx_x, vtx_y = ObtainData(stree, btree, beamspot_coordinates, IsSignal)
    total_events = len(px)

    # Obtain max data
    Smaxpx, Smaxpy, Smaxpt, Smax_dxy_err = GetMaxVar(stree['vtx_tk_px'].array()), GetMaxVar(stree['vtx_tk_py'].array()), \
                                           GetMaxVar(stree['vtx_tk_pt'].array()), GetMaxVar(
        stree['vtx_tk_dxyerr'].array())
    Bmaxpx, Bmaxpy, Bmaxpt, Bmax_dxy_err = GetMaxVar(btree['vtx_tk_px'].array()), GetMaxVar(btree['vtx_tk_py'].array()), \
                                           GetMaxVar(btree['vtx_tk_pt'].array()), GetMaxVar(
        btree['vtx_tk_dxyerr'].array())
    maxpx, maxpy, maxpt, max_dxy_err = max(Smaxpx, Bmaxpx), max(Smaxpy, Bmaxpy), max(Smaxpt, Bmaxpt), max(Smax_dxy_err,
                                                                                                          Bmax_dxy_err)

    event_number = 1
    for event in range(len(px)):
        for vertex in range(len(px[event])):
            # Obtain vertex data
            try:
                px0 = px[event][vertex]
                py0 = py[event][vertex]
                dxy0 = dxy[event][vertex]
                vtx_x0 = vtx_x[event][vertex]
                vtx_y0 = vtx_y[event][vertex]
            except ValueError:
                if IsSignal == True:
                    print("There is no data in the signal file for event number " + str(event))
                    return
                else:
                    print("There is no data in the background file for event number " + str(event))
                    return

            # Calculate secondary dxy
            tk2_dx, tk2_dy = calculateSecondaryDXY(px0, py0, dxy0, vtx_x0, vtx_y0)

            # Create image
            center = int(w / 2)
            x_initial_list, y_initial_list, x_final_list, y_final_list = calculateXYImage(px0, py0, maxpx, maxpy, tk2_dx,
                                                                                          tk2_dy, center, 4 / 5, 1)

            # Rotate components so that up direction points away from primary vertex
            vtx_theta = calculateAngle(vtx_x0, vtx_y0)
            x_final_list_rotated, y_final_list_rotated = RotateComponents(x_initial_list, y_initial_list, x_final_list,
                                                                          y_final_list, vtx_theta)

            # Calculate the angle of the rotated track
            for track in range(len(x_initial_list)):
                tk_theta = calculateAngle(x_final_list_rotated[track] - x_initial_list[track],
                                          y_initial_list[track] - y_final_list_rotated[track])
                if tk_theta > 0 and tk_theta < 90:
                    quadrants.append(1)
                elif tk_theta > 90 and tk_theta < 180:
                    quadrants.append(2)
                elif tk_theta > 180 and tk_theta < 270:
                    quadrants.append(3)
                elif tk_theta > 270 and tk_theta < 360:
                    quadrants.append(4)
                else:
                    pass

        if event_number%100 == 0:
            print("The track quadrants up to event {} have been counted for {}. {}/{} of the way done.".format(event_number, data, event_number, total_events))
        event_number+=1

    return quadrants


def PlotQuadrantBars(signal_quadrants, background_quadrants):
    #Calculate values for errorbar
    expected_signal_counts = len(signal_quadrants)/4
    expected_background_counts = len(background_quadrants)/4

    sq1, sq2, sq3, sq4 = signal_quadrants.count(1), signal_quadrants.count(2), signal_quadrants.count(3), signal_quadrants.count(4)
    bq1, bq2, bq3, bq4 = background_quadrants.count(1), background_quadrants.count(2), background_quadrants.count(3), background_quadrants.count(4)

    #Create the plots
    axes = "Quadrant Counts for Signal and Background"
    plt.figure(axes)
    plt.suptitle(axes)

    plt.subplot(211)
    plt.bar(1, sq1, label='Signal', color='Blue', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(2, sq2, color='Blue', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(3, sq3, color='Blue', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(4, sq4, color='Blue', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')

    plt.errorbar(1, expected_signal_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(2, expected_signal_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(3, expected_signal_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(4, expected_signal_counts, 0, ecolor='lightcoral', capsize=10.0)

    plt.ylabel('Counts')
    plt.legend()

    plt.subplot(212)
    plt.bar(1, bq1, label='Background', color='Orange', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(2, bq2, color='Orange', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(3, bq3, color='Orange', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')
    plt.bar(4, bq4, color='Orange', alpha=0.5, edgecolor='Black', linewidth=1.2, align='center')

    plt.errorbar(1, expected_background_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(2, expected_background_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(3, expected_background_counts, 0, ecolor='lightcoral', capsize=10.0)
    plt.errorbar(4, expected_background_counts, 0, ecolor='lightcoral', capsize=10.0)

    plt.xlabel('Quadrants')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()


def plotVertexThetaHistogram(stree, btree, beamspot_coordinates):
    #Obtain vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, beamspot_coordinates, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, beamspot_coordinates, False)

    #Flatten vertex position data to 1D list
    L_svtx_x = awk.flatten(svtx_x, axis=None)
    L_svtx_y = awk.flatten(svtx_y, axis=None)
    L_bvtx_x = awk.flatten(bvtx_x, axis=None)
    L_bvtx_y = awk.flatten(bvtx_y, axis=None)

    #Calculate vertex thetas
    s_vtx_theta = []
    b_vtx_theta = []

    for i in range(len(L_svtx_x)):
        s_vtx_theta.append(calculateAngle(L_svtx_x[i], L_svtx_y[i]))
    for i in range(len(L_bvtx_x)):
        b_vtx_theta.append(calculateAngle(L_bvtx_x[i], L_bvtx_y[i]))

    #Plot the histogram
    axes = "Vertex Theta Histogram"
    plt.figure(axes)
    plt.hist(s_vtx_theta, bins=100, label='Signal', color='blue', alpha=0.5, density=True)
    plt.hist(b_vtx_theta, bins=100, label='Background', color='orange', alpha=0.5, density=True)
    plt.suptitle(axes)
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plotTkThetaHistogram(stree, btree, beamspot_coordinates, rotation_constant=180, addVtxTheta=True, printInformation=False):
    # Obtain track and vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, beamspot_coordinates, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, beamspot_coordinates, False)

    #Flatten track momentum data to 1D list
    L_spx = awk.flatten(spx, axis=None)
    L_spy = awk.flatten(spy, axis=None)
    L_bpx = awk.flatten(bpx, axis=None)
    L_bpy = awk.flatten(bpy, axis=None)

    # Calculate vertex thetas
    s_ptheta = []
    b_ptheta = []

    for i in range(len(L_spx)):
        s_ptheta.append(calculateAngle(L_spx[i], L_spy[i]))
    for i in range(len(L_bpx)):
        b_ptheta.append(calculateAngle(L_bpx[i], L_bpy[i]))

    if addVtxTheta:
        #Flatten vertex position data to 1D list
        L_svtx_x = awk.flatten(svtx_x, axis=None)
        L_svtx_y = awk.flatten(svtx_y, axis=None)
        L_bvtx_x = awk.flatten(bvtx_x, axis=None)
        L_bvtx_y = awk.flatten(bvtx_y, axis=None)

        # Calculate vertex thetas
        s_vtx_theta = []
        b_vtx_theta = []

        for i in range(len(L_svtx_x)):
            s_vtx_theta.append(calculateAngle(L_svtx_x[i], L_svtx_y[i]))
        for i in range(len(L_bvtx_x)):
            b_vtx_theta.append(calculateAngle(L_bvtx_x[i], L_bvtx_y[i]))

        #Add vertex theta to track theta, plus a constant
        s_p_rotatedTheta = []
        b_p_rotatedTheta = []

        vertex_counter = 0
        track_counter = 0
        for event in range(len(spx)):
            for vertex in range(len(spx[event])):
                for track in range(len(spx[event][vertex])):
                    s_p_rotatedTheta.append(s_ptheta[track_counter] + s_vtx_theta[vertex_counter] + rotation_constant)
                    if printInformation == True:
                        if track_counter % 100:
                            print("Tk Theta Expected = {}".format(calculateAngle(spx[event][vertex][track], spy[event][vertex][track])))
                            print("Tk Theta Actual = {}".format(s_ptheta[track_counter]))
                            print("Vtx Theta Expected = {}".format(calculateAngle(svtx_x[event][vertex], svtx_y[event][vertex])))
                            print("Vtx Theta Actual = {}".format(s_vtx_theta[vertex_counter]))
                            print("Rotated Theta = {}".format(s_ptheta[track_counter] + s_vtx_theta[vertex_counter] + rotation_constant))
                    track_counter+=1
                vertex_counter+=1

        vertex_counter = 0
        track_counter = 0
        for event in range(len(bpx)):
            for vertex in range(len(bpx[event])):
                for track in range(len(bpx[event][vertex])):
                    b_p_rotatedTheta.append(b_ptheta[track_counter] + b_vtx_theta[vertex_counter] + rotation_constant)
                    track_counter+=1
                vertex_counter+=1

        #Reduce the range of theta to 0-360
        for i in range(len(s_p_rotatedTheta)):
            reduced_theta = s_p_rotatedTheta[i]
            while reduced_theta > 360:
                reduced_theta -= 360
            s_p_rotatedTheta[i] = reduced_theta

        for i in range(len(b_p_rotatedTheta)):
            reduced_theta = b_p_rotatedTheta[i]
            while reduced_theta > 360:
                reduced_theta -= 360
            b_p_rotatedTheta[i] = reduced_theta
    else:
        #Add rotation constant to Tk theta
        for i in range(len(s_ptheta)):
            s_ptheta[i] += rotation_constant
        for i in range(len(b_ptheta)):
            b_ptheta[i] += rotation_constant

        #Reduce the range of theta to 0-360
        for i in range(len(s_ptheta)):
            reduced_theta = s_ptheta[i]
            while reduced_theta > 360:
                reduced_theta -= 360
            s_ptheta[i] = reduced_theta

        for i in range(len(b_ptheta)):
            reduced_theta = b_ptheta[i]
            while reduced_theta > 360:
                reduced_theta -= 360
            b_ptheta[i] = reduced_theta

    # Plot the histogram
    if addVtxTheta:
        axes = "Rotated Track Theta Histogram"
    else:
        axes = "Track Theta Histogram"
    plt.figure(axes)

    if addVtxTheta:
        plt.hist(s_p_rotatedTheta, bins=100, label='Signal', color='blue', alpha=0.5, density=True)
        plt.hist(b_p_rotatedTheta, bins=100, label='Background', color='orange', alpha=0.5, density=True)
    else:
        plt.hist(s_ptheta, bins=100, label='Signal', color='blue', alpha=0.5, density=True)
        plt.hist(b_ptheta, bins=100, label='Background', color='orange', alpha=0.5, density=True)

    plt.suptitle(axes)
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def plotVtxX_VtxY(stree, btree, beamspot_coordinates):
    # Obtain track and vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, beamspot_coordinates, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, beamspot_coordinates, False)

    #Flatten vertex position data to 1D list
    L_svtx_x = awk.flatten(svtx_x, axis=None)
    L_svtx_y = awk.flatten(svtx_y, axis=None)
    L_bvtx_x = awk.flatten(bvtx_x, axis=None)
    L_bvtx_y = awk.flatten(bvtx_y, axis=None)

    # Plot the histogram
    axes = "Vertex X & Y Histograms"
    plt.figure(axes)

    plt.subplot(211)
    plt.hist(L_svtx_x, bins=50, label='Signal', color='blue', alpha=0.5, density=True)
    plt.hist(L_bvtx_x, bins=50, label='Background', color='orange', alpha=0.5, density=True)
    plt.xlabel('Primary to Secondary Vertex Displacement in X [cm]')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(212)
    plt.hist(L_svtx_y, bins=50, label='Signal', color='blue', alpha=0.5, density=True)
    plt.hist(L_bvtx_y, bins=50, label='Background', color='orange', alpha=0.5, density=True)
    plt.xlabel('Primary to Secondary Vertex Displacement in Y [cm]')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout(pad=1.25)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(axes)
    plt.show()


def plot2DVtxX_VtxY(stree, btree, beamspot_coordinates):
    # Obtain track and vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, beamspot_coordinates, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, beamspot_coordinates, False)

    # Flatten vertex position data to 1D list
    L_svtx_x = np.array(awk.flatten(svtx_x, axis=None))
    L_svtx_y = np.array(awk.flatten(svtx_y, axis=None))
    L_bvtx_x = np.array(awk.flatten(bvtx_x, axis=None))
    L_bvtx_y = np.array(awk.flatten(bvtx_y, axis=None))

    # Initiate the ranges
    xmin = min(min(L_svtx_x), min(L_bvtx_x))
    xmax = max(max(L_svtx_x), max(L_bvtx_x))
    ymin = min(min(L_svtx_y), min(L_bvtx_y))
    ymax = max(max(L_svtx_y), max(L_bvtx_y))

    # Plot the 2D histograms
    axes = "2D Vertex X & Y Histograms"
    plt.figure(axes)

    plt.subplot(211)
    h1 = plt.hist2d(x=L_svtx_x, y=L_svtx_y, bins=50, cmap='Blues', alpha=0.75)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    cbar1 = plt.colorbar(h1[3])
    cbar1.ax.get_yaxis().labelpad = 15
    cbar1.ax.set_ylabel('Signal', rotation=270)
    plt.ylabel("Vertex Y [cm]")
    vmin, vmax = plt.gci().get_clim()

    plt.subplot(212)
    h2 = plt.hist2d(x=L_bvtx_x, y=L_bvtx_y, bins=50, cmap='Oranges', alpha=0.75)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.clim(vmin, vmax)
    cbar2 = plt.colorbar(h2[3])
    cbar2.ax.get_yaxis().labelpad = 15
    cbar2.ax.set_ylabel('Background', rotation=270)
    plt.xlabel("Vertex X [cm]")
    plt.ylabel("Vertex Y [cm]")

    plt.suptitle(axes)
    plt.show()


def files_from_directory_to_list(directory, signalflag='splitSUSY'): #Combines signal files and background files into seperate lists from one directory
    #Place signal and background file directories into lists
    signalfiles = []
    bkgfiles = []
    for file in os.listdir(directory):
        if signalflag in file:
            signalfiles.append(file)
        else:
            bkgfiles.append(file)

    for i in range(len(signalfiles)):
        signalfiles[i] = os.path.join(directory, signalfiles[i])
    for i in range(len(bkgfiles)):
        bkgfiles[i] = os.path.join(directory, bkgfiles[i])

    #Remove files with zero contents
    for i in range(len(signalfiles)):
        r = uproot.open(signalfiles[i])
        try:
            r['mfvVertexTreer']['tree_DV']['vtx_tk_px'].array()
        except ValueError:
            del signalfiles[i]
    for i in range(len(bkgfiles)):
        r = uproot.open(bkgfiles[i])
        try:
            r['mfvVertexTreer']['tree_DV']['vtx_tk_px'].array()
        except ValueError:
            del bkgfiles[i]

    return signalfiles, bkgfiles


def obtain_max_data_for_directory(signaldirectory, bkgdirectory): #Obtains the max variables used in the image creation for a given set of root files in a directory
    maxvars = {'Smaxpx': [], 'Smaxpy': [], 'Smaxpt': [], 'Smax_dxy_err': [],
               'Bmaxpx': [], 'Bmaxpy': [], 'Bmaxpt': [], 'Bmax_dxy_err': []}
    for file in signaldirectory:
        sroot = uproot.open(file)
        stree = sroot['mfvVertexTreer']['tree_DV']
        maxvars['Smaxpx'].append(GetMaxVar(stree['vtx_tk_px'].array()))
        maxvars['Smaxpy'].append(GetMaxVar(stree['vtx_tk_py'].array()))
        maxvars['Smaxpt'].append(GetMaxVar(stree['vtx_tk_pt'].array()))
        maxvars['Smax_dxy_err'].append(GetMaxVar(stree['vtx_tk_dxyerr'].array()))

    for file in bkgdirectory:
        broot = uproot.open(file)
        btree = broot['mfvVertexTreer']['tree_DV']
        maxvars['Bmaxpx'].append(GetMaxVar(btree['vtx_tk_px'].array()))
        maxvars['Bmaxpy'].append(GetMaxVar(btree['vtx_tk_py'].array()))
        maxvars['Bmaxpt'].append(GetMaxVar(btree['vtx_tk_pt'].array()))
        maxvars['Bmax_dxy_err'].append(GetMaxVar(btree['vtx_tk_dxyerr'].array()))

    smaxpx, smaxpy, smaxpt, smax_dxy_err = max(maxvars['Smaxpx']), max(maxvars['Smaxpy']), max(maxvars['Smaxpt']), max(maxvars['Smax_dxy_err'])
    bmaxpx, bmaxpy, bmaxpt, bmax_dxy_err = max(maxvars['Bmaxpx']), max(maxvars['Bmaxpy']), max(maxvars['Bmaxpt']), max(maxvars['Bmax_dxy_err'])

    return {"maxpx": max(smaxpx, bmaxpx), "maxpy": max(smaxpy, bmaxpy), "maxpt": max(smaxpt, bmaxpt), "max_dxy_err": max(smax_dxy_err, bmax_dxy_err)}