import math as m
import uproot
from PIL import Image, ImageDraw

def drawLine(shape, img, tkdxy_err, max_dxy_err): #Draws a line whose magnitude is the magnitude of px and py
    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(200 * tkdxy_err/max_dxy_err)

    #Draws the line
    img.line(xy=shape, fill=grayscale, width=1)


def drawTriangle(lineshape, img, imagecenter, tkdxy_err, max_dxy_err, tk_pt, max_tk_pt): #Draw isoceles triangle where magnitude of the track is the height
    line_magnitude = m.sqrt((lineshape[1][0]-lineshape[0][0])**2 + (lineshape[1][1]-lineshape[0][1])**2)

    #Calculate magnitude of the base based on the pt of the track
    base_magnitude = (line_magnitude/3) * m.log(tk_pt)/m.log(max_tk_pt)

    #Calculate corners of the triangle
    xcorner, ycorner = base_magnitude/m.sqrt(2), base_magnitude/m.sqrt(2)
    initial_points = (lineshape[0][0], lineshape[0][1])
    if (lineshape[1][0] < imagecenter and lineshape[1][1] < imagecenter) or (lineshape[1][0] > imagecenter and lineshape[1][1] > imagecenter):
        corner1 = (lineshape[1][0]+xcorner, lineshape[1][1]-ycorner)
        corner2 = (lineshape[1][0]-xcorner, lineshape[1][1]+ycorner)
    else:
        corner1 = (lineshape[1][0] - xcorner, lineshape[1][1] - ycorner)
        corner2 = (lineshape[1][0] + xcorner, lineshape[1][1] + ycorner)

    #Calculate the greyscale of the trinagle based on the tk primary dxy error
    grayscale = int(200 * tkdxy_err/max_dxy_err)

    #Draw triangle
    img.polygon(xy=[initial_points, corner1, corner2], fill=None, outline=grayscale)


def calculateAngle(x, y): #Returns the 360 degree angle for a given 2D vector
    try:
        theta = m.degrees(m.atan(y/x))
        abstheta = abs(theta)
        if x>0 and y>0:
            pass
        elif x<0 and y>0:
            abstheta += 90.0
        elif x<0 and y<0:
            abstheta += 180.0
        elif x>0 and y<0:
            abstheta += 270.0
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


def calculateXYImage(px, py, maxpx, maxpy, tk2_dx, tk2_dy, imagecenter, scale_multiplicity_constant): #Calculate the endpoints of the track lines for the images in x y space
    #Initialize the lists
    x_initial_list = []
    y_initial_list = []
    x_final_list = []
    y_final_list = []

    #Create track magnitude constant
    cx, cy = scale_multiplicity_constant*imagecenter/m.log(maxpx), scale_multiplicity_constant*imagecenter/m.log(maxpy)

    #Resize the px and py components of the track in accordance to the constant and the log of px/py
    for i in range(len(px)):
        try:
            clogx = cx*m.log(px[i])
        except ValueError:
            clogx = -1*cx*m.log(abs(px[i]))
        try:
            clogy = cy*m.log(py[i])
        except ValueError:
            clogy = -1*cy*m.log(abs(py[i]))
        try:
            clogdx = m.log(tk2_dx[i])
        except ValueError:
            clogdx = -1*m.log(abs(tk2_dx[i]))
        try:
            clogdy = m.log(tk2_dy[i])
        except ValueError:
            clogdy = -1*m.log(abs(tk2_dy[i]))
        x_initial_list.append(round(imagecenter+clogdx))
        y_initial_list.append(round(imagecenter+clogdy))
        x_final_list.append(round(imagecenter+clogx))
        y_final_list.append(round(imagecenter+clogy))

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
        tk2_dxy.append(tk_dxy[i] - m.sin(m.radians(tkp_theta[i]-vtx_theta))*vtx_mag)
        if (tkp_theta[i] >= 0 and tkp_theta[i] <= 90) or (tkp_theta[i] > 180 and tkp_theta[i] <= 270):
            omega = 90 - tkp_theta[i]
        else:
            omega = tkp_theta[i] - 90
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
print(stree.keys())

#Obtain momentum data
px = stree['vtx_tk_px'].array()
py = stree['vtx_tk_py'].array()
pt = stree['vtx_tk_pt'].array()
dxy = stree['vtx_tk_dxy'].array()
dxy_err = stree['vtx_tk_dxyerr'].array()
vtx_x = stree['vtx_x'].array()
vtx_y = stree['vtx_y'].array()

#Obtain max data
maxpx, maxpy, maxpt, max_dxy_err = GetMaxVar(px), GetMaxVar(py), GetMaxVar(pt), GetMaxVar(dxy_err)

#Obtain first vertex data
px0 = px[0][0]
py0 = py[0][0]
pt0 = pt[0][0]
dxy0 = dxy[0][0]
vtx_x0 = vtx_x[0][0]
vtx_y0 = vtx_y[0][0]
dxy_err0 = dxy_err[0][0]

#Calculate secondary dxy
tk2_dx, tk2_dy = calculateSecondaryDXY(px0, py0, dxy0, vtx_x0, vtx_y0)

#Create image
w, h = 224, 224
center = int(w/2)
x_initial_list, y_initial_list, x_final_list, y_final_list = calculateXYImage(px0, py0, maxpx, maxpy, tk2_dx, tk2_dy, center, 2/3)
img = Image.new(mode='L', size=(w, h), color=255)
img1 = ImageDraw.Draw(img)

#Rotate components so that up direction points away from primary vertex
vtx_theta = calculateAngle(vtx_x0, vtx_y0)
x_final_list_rotated, y_final_list_rotated = RotateComponents(x_initial_list, y_initial_list, x_final_list, y_final_list, vtx_theta)

#Draw Triangles
for i in range(len(x_final_list)):
    #shape = [(x_initial_list[i], y_initial_list[i]), (x_final_list[i], y_final_list[i])]
    shape = [(x_initial_list[i], y_initial_list[i]), (x_final_list_rotated[i], y_final_list_rotated[i])]
    drawTriangle(shape, img1, center, dxy_err0[i], max_dxy_err, pt0[i], maxpt)
    #drawLine(shape, img1, dxy_err0[i], max_dxy_err)
img.show()