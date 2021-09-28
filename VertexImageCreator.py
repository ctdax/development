import math as m
import uproot
import sys
from PIL import Image, ImageDraw

def drawLine(shape, img, tkdxy_err, max_dxy_err): #Draws a line whose magnitude is the magnitude of px and py
    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(200 * tkdxy_err/max_dxy_err)

    #Draws the line
    img.line(xy=shape, fill=grayscale, width=1)


def drawTriangle(lineshape, img, imagecenter, tkdxy_err, max_dxy_err, tk_pt, max_tk_pt): #Draw isoceles triangle where magnitude of the track is the height
    line_magnitude = m.sqrt((lineshape[1][0]-lineshape[0][0])**2 + (lineshape[1][1]-lineshape[0][1])**2)

    #Calculate magnitude of the base based on the pt of the track
    base_magnitude = (line_magnitude/2) * m.log(tk_pt)/m.log(max_tk_pt)

    #Calculate the angle of the rotated track
    tk_theta = calculateAngle(lineshape[1][0]-lineshape[0][0], lineshape[0][1]-lineshape[1][1])

    #Calculate corners of the triangle
    calculation_tk_theta = tk_theta
    while calculation_tk_theta > 90:
        calculation_tk_theta -= 90
    xcorner, ycorner = base_magnitude*m.sin(m.radians(tk_theta)), base_magnitude*m.cos(m.radians(tk_theta))
    initial_points = (lineshape[0][0], lineshape[0][1])

    if (90 < tk_theta and tk_theta <= 180) or (270 < tk_theta and tk_theta < 360):
        corner1 = (lineshape[1][0]-xcorner, lineshape[1][1]-ycorner)
        corner2 = (lineshape[1][0]+xcorner, lineshape[1][1]+ycorner)
    else:
        corner1 = (lineshape[1][0] - xcorner, lineshape[1][1] - ycorner)
        corner2 = (lineshape[1][0] + xcorner, lineshape[1][1] + ycorner)

    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(200 * tkdxy_err/max_dxy_err)

    #Draw triangle
    img.polygon(xy=[initial_points, corner1, corner2], fill=None, outline=grayscale)


def ObtainData(IsSignal=True):
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
        theta = m.degrees(m.atan(y/x))
        abstheta = abs(theta)
        if x>0 and y>0:
            pass
        elif x<0 and y>0:
            abstheta = 180.0 - abstheta
        elif x<0 and y<0:
            abstheta += 180.0
        elif x>0 and y<0:
            abstheta = 360 - abstheta
        elif x>0 and y==0:
            abstheta = 0.0
        elif x<0 and y==0:
            abstheta = 180.0
        round(abstheta, 2)
    except ZeroDivisionError:
        if y>0:
            abstheta = 90.0
        else:
            abstheta = 270.0

    return abstheta


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
        y_initial_list.append(round(imagecenter+clogdy))
        x_final_list.append(round(x_initial_list[i]+clogx))
        y_final_list.append(round(y_initial_list[i]+clogy))

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
        deltay = y_final_list[i]-y_initial_list[i]
        rotation_angle = vtx_theta + 180
        x_final_list_rotated.append(deltax*m.cos(m.radians(rotation_angle)) - deltay*m.sin(m.radians(rotation_angle)))
        y_final_list_rotated.append(deltax * m.sin(m.radians(rotation_angle)) + deltay * m.cos(m.radians(rotation_angle)))
        x_final_list_rotated_fixed.append(x_initial_list[i] + x_final_list_rotated[i])
        y_final_list_rotated_fixed.append(y_initial_list[i] + y_final_list_rotated[i])

    return x_final_list_rotated_fixed, y_final_list_rotated_fixed


#Import ROOT files
sigrootfile = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\roots\splitSUSY_tau000001000um_M2000_1800_2017_vertextree.root"
bkgrootfile = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\roots\ttbar_2017_vertextree.root"
sfile = uproot.open(sigrootfile)
bfile = uproot.open(bkgrootfile)
stree = sfile['mfvVertexTreer']['tree_DV']
btree = bfile['mfvVertexTreer']['tree_DV']

#Obtain momentum data
px, py, pt, dxy, dxy_err, vtx_x, vtx_y = ObtainData(IsSignal=True)

#Obtain max data
Smaxpx, Smaxpy, Smaxpt, Smax_dxy_err = GetMaxVar(stree['vtx_tk_px'].array()), GetMaxVar(stree['vtx_tk_py'].array()),\
                                       GetMaxVar(stree['vtx_tk_pt'].array()), GetMaxVar(stree['vtx_tk_dxyerr'].array())
Bmaxpx, Bmaxpy, Bmaxpt, Bmax_dxy_err = GetMaxVar(btree['vtx_tk_px'].array()), GetMaxVar(btree['vtx_tk_py'].array()),\
                                       GetMaxVar(btree['vtx_tk_pt'].array()), GetMaxVar(btree['vtx_tk_dxyerr'].array())
maxpx, maxpy, maxpt, max_dxy_err = max(Smaxpx, Bmaxpx), max(Smaxpy, Bmaxpy), max(Smaxpt, Bmaxpt), max(Smax_dxy_err, Bmax_dxy_err)

#Obtain first vertex data
event, vertex = 0, 0
px0 = px[event][vertex]
py0 = py[event][vertex]
pt0 = pt[event][vertex]
dxy0 = dxy[event][vertex]
vtx_x0 = vtx_x[event][vertex]
vtx_y0 = vtx_y[event][vertex]
dxy_err0 = dxy_err[event][vertex]

#Calculate secondary dxy
tk2_dx, tk2_dy = calculateSecondaryDXY(px0, py0, dxy0, vtx_x0, vtx_y0)

#Create image
w, h = 224, 224
center = int(w/2)
x_initial_list, y_initial_list, x_final_list, y_final_list = calculateXYImage(px0, py0, maxpx, maxpy, tk2_dx, tk2_dy, center, 4/5, 2)
img = Image.new(mode='L', size=(w, h), color=255)
img1 = ImageDraw.Draw(img)

#Draw circle at image center to represent secondary vertex
img1.ellipse((center-2,center-2,center+2,center+2), fill=None, outline=0)

#Rotate components so that up direction points away from primary vertex
vtx_theta = calculateAngle(vtx_x0, vtx_y0)
x_final_list_rotated, y_final_list_rotated = RotateComponents(x_initial_list, y_initial_list, x_final_list, y_final_list, vtx_theta)

#Draw Triangles
for i in range(len(x_final_list)):
    shape = [(x_initial_list[i], y_initial_list[i]), (x_final_list_rotated[i], y_final_list_rotated[i])]
    #drawTriangle(shape, img1, center, dxy_err0[i], max_dxy_err, pt0[i], maxpt)
    drawLine(shape, img1, dxy_err0[i], max_dxy_err)
img.show()