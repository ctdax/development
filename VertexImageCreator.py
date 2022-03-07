import uproot
import os
from VertexImageCreatorFunctions import createVertexImage, Observe_N_Vertices, CountTrackQuadrants, PlotQuadrantBars,\
    plotVertexThetaHistogram, plotTkThetaHistogram, plotVtxX_VtxY, plot2DVtxX_VtxY, files_from_directory_to_list, obtain_max_data_for_directory

b'Import ROOT files'
beamspot_coordinates = [-0.02479, 0.06929, 0.7899] #X, Y, Z
image_filepath = os.path.join(os.path.expanduser('~'), 'Box Sync', 'Neu-work', 'Longlive master', 'ML Vertex Images')
root_directory = os.path.join(os.path.expanduser('~'), 'Box Sync', 'Neu-work', 'Longlive master', 'roots', 'vertextree')

signalfiles, bkgfiles = files_from_directory_to_list(root_directory)
maxdict = obtain_max_data_for_directory(signalfiles, bkgfiles)

#for file in signalfiles:
#    print(file)
#    btree = None
#    sroot = uproot.open(file)
#    stree = sroot['mfvVertexTreer']['tree_DV']
#    Observe_N_Vertices('all', stree, btree, beamspot_coordinates, w=32, h=32, ImageFilePath=image_filepath, showTitle=False,
#                       drawCenter=False, maxData=maxdict, multipleRoots=True, rootFileName=file)

#for file in bkgfiles:
#    print(file)
#    stree = None
#    broot = uproot.open(file)
#    btree = broot['mfvVertexTreer']['tree_DV']
#    Observe_N_Vertices('all', stree, btree, beamspot_coordinates, w=32, h=32, ImageFilePath=image_filepath, showTitle=False,
#                       drawCenter=False, maxData=maxdict, multipleRoots=True, rootFileName=file)

sroot = uproot.open(signalfiles[0])
broot = uproot.open(bkgfiles[0])
stree = sroot['mfvVertexTreer']['tree_DV']
btree = broot['mfvVertexTreer']['tree_DV']

b'Run Individual Vertex'
#createVertexImage(stree, btree, beamspot_coordinates, event=0, vertex=0, w=32, h=32, ImageFilePath=image_filepath,
#                  IsSignal=True, showTitle=False, drawCenter=False)

b'Run N Vertices'
if __name__ == '__main__':
    Observe_N_Vertices(20, stree, btree, beamspot_coordinates, w=256, h=256, IsSignal=True, ImageFilePath=image_filepath,
                       showTitle=False, drawCenter=False, maxData=maxdict, rootFileName=signalfiles[0], parallel=True, start=10)

b'Quadrant Analysis'
#signal_quadrants = CountTrackQuadrants(stree, btree, beamspot_coordinates, IsSignal=True)
#background_quadrants = CountTrackQuadrants(stree, btree, beamspot_coordinates, IsSignal=False)
#PlotQuadrantBars(signal_quadrants, background_quadrants)

b'Vertex angle, track angle, and vertex position analyses'
#plotVertexThetaHistogram(stree, btree, beamspot_coordinates)
#plotTkThetaHistogram(stree, btree, beamspot_coordinates, rotation_constant=180, addVtxTheta=True, printInformation=False)
#plotVtxX_VtxY(stree, btree, beamspot_coordinates)
#plot2DVtxX_VtxY(stree, btree, beamspot_coordinates)