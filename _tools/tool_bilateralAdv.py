import os

################ MAKE CHANGES HERE #################
inputFileFormat = "%03d"            # name of input files, e.g., %03d if files are named 001.png, 002.png
imageFormat   = "input\\" + inputFileFormat + ".png"
flowFwdFormat = "flow_fwd\\" + inputFileFormat + ".A2V2f"  # path to the forward flow files (computed by _tools/disflow)
flowBwdFormat = "flow_bwd\\" + inputFileFormat + ".A2V2f"  # path to the backward flow files (computed by _tools/disflow)
outputFormat   = "input_filtered\\" + inputFileFormat + ".png"  # path to the result filtered sequence
frameFirst = 1                      # number of the first PNG file in the input folder
frameLast = 109                     # number of the last PNG file in the input folder
####################################################

firstFrame = frameFirst
lastFrame= frameLast  
frameStep = +1

os.makedirs(os.path.dirname(outputFormat),exist_ok=True)

for frame in range(firstFrame,lastFrame+frameStep,frameStep):  	
  filter = "bilateralAdv.exe "+imageFormat+" "+flowFwdFormat+" "+flowBwdFormat+(" %d "%(frame))+" 15 16 "+(outputFormat%(frame))
  #print(filter)
  os.system(filter)
