import math as m
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import awkward as awk

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


def ObtainData(stree, btree, IsSignal=True):
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
        rotation_angle = 0
        x_final_list_rotated.append(deltax*m.cos(m.radians(rotation_angle)) - deltay*m.sin(m.radians(rotation_angle)))
        y_final_list_rotated.append(deltax * m.sin(m.radians(rotation_angle)) + deltay * m.cos(m.radians(rotation_angle)))
        x_final_list_rotated_fixed.append(x_initial_list[i] + x_final_list_rotated[i])
        y_final_list_rotated_fixed.append(y_initial_list[i] + y_final_list_rotated[i])

        #print("Vertex theta = {}".format(vtx_theta))
        #print("Track theta = {}".format(calculateAngle(deltax, deltay)))
        #print("Final expected theta = {}".format(calculateAngle(deltax, deltay) + rotation_angle))
        #print("X = {}".format(deltax))
        #print("X' rotated = {}".format(x_final_list_rotated[i]))
        #print("Y = {}".format(deltay))
        #print("Y' rotated = {}".format(y_final_list_rotated[i]))
        #print("Final actual theta = {}".format(calculateAngle(x_final_list_rotated[i], y_final_list_rotated[i])))
        #print("X' initial = {}".format(x_initial_list[i]))
        #print("X' final = {}".format(x_final_list_rotated_fixed[i]))
        #print("Y' initial = {}".format(y_initial_list[i]))
        #print("Y' final = {}".format(y_final_list_rotated_fixed[i]))

    return x_final_list_rotated_fixed, y_final_list_rotated_fixed


def createVertexImage(stree, btree, event, vertex, ImageFilePath=None, IsSignal=True, showTitle=False):
    # Obtain momentum data
    px, py, pt, dxy, dxy_err, vtx_x, vtx_y = ObtainData(stree, btree, IsSignal)

    # Obtain max data
    Smaxpx, Smaxpy, Smaxpt, Smax_dxy_err = GetMaxVar(stree['vtx_tk_px'].array()), GetMaxVar(stree['vtx_tk_py'].array()), \
                                           GetMaxVar(stree['vtx_tk_pt'].array()), GetMaxVar(stree['vtx_tk_dxyerr'].array())
    Bmaxpx, Bmaxpy, Bmaxpt, Bmax_dxy_err = GetMaxVar(btree['vtx_tk_px'].array()), GetMaxVar(btree['vtx_tk_py'].array()), \
                                           GetMaxVar(btree['vtx_tk_pt'].array()), GetMaxVar(btree['vtx_tk_dxyerr'].array())
    maxpx, maxpy, maxpt, max_dxy_err = max(Smaxpx, Bmaxpx), max(Smaxpy, Bmaxpy), max(Smaxpt, Bmaxpt), max(Smax_dxy_err,
                                                                                                          Bmax_dxy_err)

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
    w, h = 224, 224
    center = int(w / 2)
    x_initial_list, y_initial_list, x_final_list, y_final_list = calculateXYImage(px0, py0, maxpx, maxpy, tk2_dx,
                                                                                  tk2_dy, center, 4 / 5, 1)
    img = Image.new(mode='L', size=(w, h), color=255)
    img1 = ImageDraw.Draw(img)

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
        # drawLine(shape, img1, dxy_err0[i], max_dxy_err)
    if ImageFilePath != None:
        if IsSignal == True:
            filepath = ImageFilePath + r"\signal\Evt {} Vtx {}.png".format(event, vertex)
        else:
            filepath = ImageFilePath + r"\background\Evt {} Vtx {}.png".format(event, vertex)
        img.save(filepath)
    else:
        img.show()


def Observe_N_Vertices(N, stree=None, btree=None, ImageFilePath=None, showTitle=False): #View or save N signal or background vertices, set N='all' to view all vertices
    #Observe N signal vertices
    if stree == None:
        pass
    else:
        if N == 'all':
            l = []
            for event in range(len(stree['vtx_tk_px'].array())):
                for vertex in range(len(stree['vtx_tk_px'].array()[event])):
                    l.append(vertex)
            N = len(l)
        n = 1
        while n <= N:
            for event in range(len(stree['vtx_tk_px'].array())):
                if n == N+1:
                    break
                for vertex in range(len(stree['vtx_tk_px'].array()[event])):
                    createVertexImage(stree, btree, event, vertex, ImageFilePath=ImageFilePath, IsSignal=True, showTitle=showTitle)
                    n+=1
                    if n == N+1:
                        break

    #Observe N background vertices
    if btree == None:
        pass
    else:
        if N == 'all':
            l = []
            for event in range(len(btree['vtx_tk_px'].array())):
                for vertex in range(len(btree['vtx_tk_px'].array()[event])):
                    l.append(vertex)
            N = len(l)
        n = 1
        while n <= N:
            for event in range(len(btree['vtx_tk_px'].array())):
                if n == N+1:
                    break
                for vertex in range(len(btree['vtx_tk_px'].array()[event])):
                    createVertexImage(stree, btree, event, vertex, ImageFilePath=ImageFilePath, IsSignal=False, showTitle=showTitle)
                    n+=1
                    if n == N+1:
                        break


def CountTrackQuadrants(stree, btree, IsSignal=True):
    #Initiate quadrant list
    quadrants = []
    if IsSignal:
        data = 'signal'
    else:
        data = 'background'

    # Obtain momentum data
    px, py, pt, dxy, dxy_err, vtx_x, vtx_y = ObtainData(stree, btree, IsSignal)
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
            w, h = 224, 224
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


def plotVertexThetaHistogram(stree, btree):
    #Obtain vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, False)

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


def plotTkThetaHistogram(stree, btree, rotation_constant=180, addVtxTheta=True):
    # Obtain track and vertex position data
    spx, spy, spt, sdxy, sdxy_err, svtx_x, svtx_y = ObtainData(stree, btree, True)
    bpx, bpy, bpt, bdxy, bdxy_err, bvtx_x, bvtx_y = ObtainData(stree, btree, False)

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