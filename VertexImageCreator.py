import uproot
from VertexImageCreatorFunctions import createVertexImage, Observe_N_Vertices, CountTrackQuadrants, PlotQuadrantBars,\
    plotVertexThetaHistogram, plotTkThetaHistogram

#Import ROOT files
sigrootfile = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\roots\splitSUSY_tau000001000um_M2000_1800_2017_vertextree.root"
bkgrootfile = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\roots\ttbar_2017_vertextree.root"
sfile = uproot.open(sigrootfile)
bfile = uproot.open(bkgrootfile)
stree = sfile['mfvVertexTreer']['tree_DV']
btree = bfile['mfvVertexTreer']['tree_DV']

filepath = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\Vertex Images"
#createVertexImage(stree, btree, event=0, vertex=0, ImageFilePath=None, IsSignal=True, showTitle=False)
#Observe_N_Vertices(20, stree, btree, ImageFilePath=filepath, showTitle=True)

#signal_quadrants = CountTrackQuadrants(stree, btree, IsSignal=True)
#background_quadrants = CountTrackQuadrants(stree, btree, IsSignal=False)
#PlotQuadrantBars(signal_quadrants, background_quadrants)

plotVertexThetaHistogram(stree, btree)
#plotTkThetaHistogram(stree, btree, rotation_constant=180, addVtxTheta=True)

