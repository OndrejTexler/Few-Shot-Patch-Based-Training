import os

################ MAKE CHANGES HERE #################
inputDir = "input"              # path to the input sequence PNGs
inputFileFormat = "%03d"        # name of input files, e.g., %03d if files are named 001.png, 002.png
inputFileExt = "png"            # extension of input files (without .), e.g., png, jpg
flowFwdDir = "flow_fwd"         # path to the output forward flow files
flowBwdDir = "flow_bwd"         # path to the output backward flow files
FIRST = 1                       # number of the first PNG file in the input folder
LAST = 109                      # number of the last PNG file in the input folder
####################################################


if not os.path.exists(flowFwdDir):
    os.mkdir(flowFwdDir)
    
if not os.path.exists(flowBwdDir):
    os.mkdir(flowBwdDir)

inputFiles = inputDir + "/" + inputFileFormat + "." + inputFileExt
flwFwdFile = flowFwdDir + "/" + inputFileFormat + ".A2V2f"
flwBwdFile = flowBwdDir + "/" + inputFileFormat + ".A2V2f"

firstFrame = FIRST+1
lastFrame  = LAST
frameStep  = +1

for frame in range(firstFrame,lastFrame+frameStep,frameStep):
  os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwFwdFile%(frame)))

firstFrame = LAST-1
lastFrame  = FIRST
frameStep  = -1

for frame in range(firstFrame,lastFrame+frameStep,frameStep):
  os.system("disflow %s %s %s"%(inputFiles%(frame),inputFiles%(frame-frameStep),flwBwdFile%(frame)))

