import math as m
import uproot
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def drawLine(shape, img):
    img.line(xy=shape, fill='black', width=1)


def drawTriangle(lineshape, img):
    line_magnitude = m.sqrt((lineshape[1][0]-lineshape[0][0])**2 + (lineshape[1][1]-lineshape[0][1])**2)
    base_magnitude = line_magnitude/4
    xcorner, ycorner = base_magnitude/m.sqrt(2), base_magnitude/m.sqrt(2)
    center = (lineshape[0][0], lineshape[0][1])
    if lineshape[1][0] < center[0] and lineshape[1][1] < center[0]:
        corner1 = (lineshape[1][0]+xcorner, lineshape[1][1]-ycorner)
        corner2 = (lineshape[1][0]-xcorner, lineshape[1][1]+ycorner)
    elif lineshape[1][0] > center[0] and lineshape[1][1] > center[0]:
        corner2 = (lineshape[1][0] + xcorner, lineshape[1][1] - ycorner)
        corner1 = (lineshape[1][0] - xcorner, lineshape[1][1] + ycorner)
    else:
        corner1 = (lineshape[1][0] - xcorner, lineshape[1][1] - ycorner)
        corner2 = (lineshape[1][0] + xcorner, lineshape[1][1] + ycorner)
    img.polygon(xy=[center, corner1, corner2], fill=None, outline=None)


def calculateAngle(x, y):
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


def GetMaxP(px, py, pz):
    maxpx = []
    maxpy = []
    maxpz = []
    for a in px:
        for b in a:
            maxpx.append(max(b))
    for a in py:
        for b in a:
            maxpy.append(max(b))
    for a in pz:
        for b in a:
            maxpz.append(max(b))

    return max(maxpx), max(maxpy), max(maxpz)


def calculateXYImage(px, py, pz, maxpx, maxpy, maxpz, tk2_dx, tk2_dy, imagecenter, scale_multiplicity_constant): #Calculate the endpoints of the track lines for the images in x y space
    xilist = []
    yilist = []
    xflist = []
    yflist = []
    cx, cy, cz = scale_multiplicity_constant*imagecenter/m.log(maxpx), scale_multiplicity_constant*imagecenter/m.log(maxpy), scale_multiplicity_constant*imagecenter/m.log(maxpz)
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
            clogz = cz*m.log(pz[i])
        except ValueError:
            clogz = -1*cz*m.log(abs(pz[i]))
        try:
            clogdx = m.log(tk2_dx[i])
        except ValueError:
            clogdx = -1*m.log(abs(tk2_dx[i]))
        try:
            clogdy = m.log(tk2_dy[i])
        except ValueError:
            clogdy = -1*m.log(abs(tk2_dy[i]))
        #xlist.append(round(imagecenter+clogx+clogdx+(clogz/m.sqrt(2))))
        #ylist.append(round(imagecenter+clogy+clogdy+(clogz/m.sqrt(2))))
        xilist.append(round(imagecenter+clogdx))
        yilist.append(round(imagecenter+clogdy))
        xflist.append(round(imagecenter+clogx))
        yflist.append(round(imagecenter+clogy))

    return xilist, yilist, xflist, yflist


def calculateSecondaryDXY(tk_px, tk_py, tk_dxy, vtx_x, vtx_y):
    vtx_mag = m.sqrt(vtx_x**2 + vtx_y**2)
    vtx_theta = calculateAngle(vtx_x, vtx_y)
    tkp_theta = []
    tk2_dxy = []
    tk2_dx = []
    tk2_dy = []
    for i in range(len(tk_px)):
        tkp_theta.append(calculateAngle(tk_px[i], tk_py[i]))
        tk2_dxy.append(tk_dxy[i] - m.sin(m.radians(tkp_theta[i]-vtx_theta))*vtx_mag)
        if (tkp_theta[i] >= 0 and tkp_theta[i] <= 90) or (tkp_theta[i] > 180 and tkp_theta[i] <= 270):
            omega = 90 - tkp_theta[i]
        else:
            omega = tkp_theta[i] - 90
        tk2_dx.append(tk2_dxy[i]*m.cos(m.radians(omega)))
        tk2_dy.append(tk2_dxy[i]*m.sin(m.radians(omega)))
    print(tk2_dxy, '\n', tk2_dx, '\n', tk2_dy)

    return tk2_dx, tk2_dy


def RotateComponents(xilist, yilist, xflist, yflist, vtx_theta):
    xflist_rotated = []
    yflist_rotated = []
    for i in range(len(xilist)):
        deltax = xflist[i]-xilist[i]
        deltay = yflist[i]-yilist[i]
        deltamag = m.sqrt(deltax**2 + deltay**2)
        tk_theta = calculateAngle(deltax, deltay)
        rotation_angle = 180 + vtx_theta + tk_theta
        while rotation_angle >= 360:
            rotation_angle -= 360

        if rotation_angle >= 0 and rotation_angle <= 90:
            xflist_rotated.append(deltamag*m.cos(m.radians(rotation_angle)))
            yflist_rotated.append(deltamag*m.sin(m.radians(rotation_angle)))
        elif rotation_angle > 90  and rotation_angle <= 180:
            rotation_angle_fixed = rotation_angle - 90
            xflist_rotated.append(-1 * deltamag * m.cos(m.radians(rotation_angle_fixed)))
            yflist_rotated.append(deltamag * m.sin(m.radians(rotation_angle_fixed)))
        elif rotation_angle > 180 and rotation_angle <= 270:
            rotation_angle_fixed = rotation_angle - 180
            xflist_rotated.append(-1 * deltamag*m.cos(m.radians(rotation_angle_fixed)))
            yflist_rotated.append(-1 * deltamag*m.sin(m.radians(rotation_angle_fixed)))
        else:
            rotation_angle_fixed = rotation_angle - 270
            xflist_rotated.append(deltamag * m.cos(m.radians(rotation_angle_fixed)))
            yflist_rotated.append(-1 * deltamag * m.sin(m.radians(rotation_angle_fixed)))

    return xflist_rotated, yflist_rotated




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
pz = stree['vtx_tk_pz'].array()
dxy = stree['vtx_tk_dxy'].array()
vtx_x = stree['vtx_x'].array()
vtx_y = stree['vtx_y'].array()
maxpx, maxpy, maxpz = GetMaxP(px, py, pz)
px0 = px[0][0]
py0 = py[0][0]
pz0 = pz[0][0]
dxy0 = dxy[0][0]
vtx_x0 = vtx_x[0][0]
vtx_y0 = vtx_y[0][0]

#Calculate secondary dxy
tk2_dx, tk2_dy = calculateSecondaryDXY(px0, py0, dxy0, vtx_x0, vtx_y0)

#Create image
w, h = 224, 224
center = int(w/2)
xilist, yilist, xflist, yflist = calculateXYImage(px0, py0, pz0, maxpx, maxpy, maxpz, tk2_dx, tk2_dy, center, 1)
img = Image.new(mode='1', size=(w, h), color=1)
img1 = ImageDraw.Draw(img)

#Rotate components so that up direction points away from primary vertex
vtx_theta = calculateAngle(vtx_x0, vtx_y0)
xflist_rotated, yflist_rotated = RotateComponents(xilist, yilist, xflist, yflist, vtx_theta)
xflist_rotated_fixed = []
yflist_rotated_fixed = []
for i in range(len(xilist)):
    xflist_rotated_fixed.append(xilist[i] + xflist_rotated[i])
    yflist_rotated_fixed.append(yilist[i] + yflist_rotated[i])

print('xilist: ', xilist, '\nxflist: ', xflist, '\nxflist_rotated: ', xflist_rotated_fixed,
      '\nyilist: ', yilist, '\nyflist: ', yflist, '\nyflist_rotated: ', yflist_rotated_fixed)

#Draw Triangles
for i in range(len(xflist)):
    shape = [(xilist[i], yilist[i]), (xflist[i], yflist[i])]
    #shape = [(xilist[i], yilist[i]), (xflist_rotated_fixed[i], yflist_rotated_fixed[i])]
    #drawTriangle(shape, img1)
    drawLine(shape, img1)
img.show()