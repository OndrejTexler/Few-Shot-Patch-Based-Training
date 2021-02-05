# Interactive Video Stylization Using Few-Shot Patch-Based Training

The official implementation of

> **Interactive Video Stylization Using Few-Shot Patch-Based Training** </br>
_[O. Texler](https://ondrejtexler.github.io/), [D. Futschik](https://dcgi.fel.cvut.cz/people/futscdav),
[M. Kučera](https://www.linkedin.com/in/kuceram/), [O. Jamriška](https://dcgi.fel.cvut.cz/people/jamriond), 
[Š. Sochorová](https://dcgi.fel.cvut.cz/people/sochosar), [M. Chai](http://www.mlchai.com), 
[S. Tulyakov](http://www.stulyakov.com), and [D. Sýkora](https://dcgi.fel.cvut.cz/home/sykorad/)_ </br>
[[`WebPage`](https://ondrejtexler.github.io/patch-based_training)],
[[`Paper`](https://ondrejtexler.github.io/res/Texler20-SIG_patch-based_training_main.pdf)],
[[`BiBTeX`](#CitingFewShotPatchBasedTraining)]

![Teaser](doc/teaser.gif)


## Run

Download the 
[testing-data.zip](https://drive.google.com/file/d/1EscSNFg4ILpB7dxr-zYw_UdOILLmDlRj/view?usp=sharing), 
and unzip. The _train folder is expected to be next to the _gen folder.

### Pre-Trained Models
If you want just quickly test the network, here are some 
[pre-trained-models.zip](https://drive.google.com/file/d/11_lCPqKDAtkMQTCSNTKBu2Sii8km_04s/view?usp=sharing).
Unzip, and follow with the Generate step. Be sure to set the correct --checkpoint path 
when calling `generate.py`, e.g., `_pre-trained-models/Zuzka2/model_00020.pth`.


### Train
To train the network, run the `train.py` 
See the example command below:

```
train.py --config "_config/reference_P.yaml" 
		 --data_root "Zuzka2_train" 
		 --log_interval 1000 
		 --log_folder logs_reference_P
```

Every 1000 (log_interval) epochs, `train.py` saves the current generator to 
logs_reference_P (log_folder), and it validates/runs the generator on _gen data - the 
result is saved in Zuzka2_gen/res__P


### Generate
To generate the results, run `generate.py`. 

```
generate.py --checkpoint "Zuzka2_train/logs_reference_P/model_00020.pth" 
	    --data_root "Zuzka2_gen"
	    --dir_input "input_filtered"
	    --outdir "Zuzka2_gen/res_00020" 
	    --device "cuda:0"
```

To generate the results on live webcam footage, run `generate_webcam.py`. To stop the generation, press q while the preview window is active.

```
generate_webcam.py --checkpoint "Zuzka2_train/logs_reference_P/model_00020.pth" 
	    --device "cuda:0"
	    --resolution 1280 720
	    --show_original 1
	    --resize 256
```
An optional resolution argument has been added, but the images will be always cropped to square, and resized to the size of resize x resize for shorter delay.


## Installation
Tested on Windows 10, `Python 3.7.8`, `CUDA 10.2`.
With the following python packages:
```
numpy                  1.19.1
opencv-python          4.4.0.40
Pillow                 7.2.0
PyYAML                 5.3.1
scikit-image           0.17.2
scipy                  1.5.2
tensorflow 	       1.15.3 (tensorflow is used only in the logger.py, I will remove this not-necessary dependency soon)
torch                  1.6.0
torchvision            0.7.0
```

## Temporal Consistency [Optional]
This section is optional. It describes steps that can help to maintain temporal 
coherency of the resulting video sequence. All example commands and build scripts in this section 
assume Windows; however, it should be really straightforward to
build it and run it on Linux/MacOS.

As the temporal consistency in our technique is not explicitly enforced, it gives us many advantages, 
e.g., parallel processing, fast training, etc., but the resulting stylized sequence may contain disturbing 
amount of flickering. While temporal consistency can be caused by various factors, below, we discuss how
to deal with two most crucial of them.


### Noise in the Input Sequence
The input video sequence captured by a camera usually contains some amount of temporal noise. 
While this noise might not be visible by the naked eye or might seem negligible, the network 
tends to amplify it. To deal with this issue, we propose to filter the input sequence using 
time-aware bilateral filter.

First, optical flow has to be computed. Use the optical flow tool in `_tools/disflow`. 
See section **Build disflow** below on how to build the tool.
Once `disflow.exe` is built and present in the PATH, see and modify the first few lines of 
`_tools/tool_disflow.py`, and run it. It reads PNGs from the `input` folder and stores optical flow in 
`flow_fwd` and `flow_bwd` folder.

Once, the optical flow is computed, use time-aware bilateral filter tool `_tools/bilateralAdv` to
filter the sequence. See section **Build bilateralAdv** below on how to build the tool.
Once `bilateralAdv.exe` is built and present in the PATH, see and modify the first few lines of 
`_tools/tool_bilateralAdv.py`, and run it. It reads PNGs from the `input` folder, and optical flow data 
from the `flow_fwd` and `flow_bwd`; it stores filtered sequence in `input_filtered`. 
Note, feel free to parallelize the for loop in `_tools/tool_bilateralAdv.py`, `bilateralAdv.exe` 
uses optical flow and can be run frame by frame independently. Also, feel free to optimize 
`bilateralAdv.exe` so that is uses multiple CPU-cores or even a GPU ... I am thrilled to see 
your pull request :-)

Finally, to do the training and inference, use filtered `input_sequence` images instead 
of the original noisy `input` images. Hopefully, the results will be more stable in time.


### Ambiguity in the Training Data
As the network is trained on small, by default 32x32 px patches, it is likely that multiple 32x32 px 
patches from `input` RGB frame will be very similar. For instance, if there is sky in the background of input 
image, patches from left and right part of the sky will likely be very similar. The problem is that in the
stylyzed exemplar, these patches might be stylized slightly differently. And that is the ambiguity, 
multiple similar input patches will be, during the training, mapped to different stylized patches. 
To deal with this, we propose to use an auxiliary RGB input images that will make all input
patches unique.

First, optical flow has to be computed. Use the optical flow tool in `_tools/disflow`. 
See section **Build disflow** below on how to build the tool.
Once `disflow.exe` is built and present in the PATH, see and modify the first few lines of 
`_tools/tool_disflow.py`, and run it. It reads PNGs from the `input` folder and stores optical flow in 
`flow_fwd` and `flow_bwd` folder.

Once, the optical flow is computed, use `_tools/gauss` to compute auxiliary gaussian mixture images.
See section **Build gauss** below on how to build the tool.
Once `gauss.exe` is built and present in the PATH, see and modify the first few lines of 
`_tools/tool_gauss.py`, and run it. It reads mask images from the `mask` folder 
(these masks can but do not need to match the masks you use during training, 
see the section below for more info), and optical flow data 
from the `flow_fwd` and `flow_bwd`; it outputs two different gaussian mixtures in 
`input_gdisko_gauss_r10_s10` (smaller circles) and `input_gdisko_gauss_r10_s15` (larger circles).
Pick one of them, e.g., `input_gdisko_gauss_r10_s10`, if it does not work well, try the other one.
Place the folder `input_gdisko_gauss_r10_s10` next to your `input` folder in both _gen as 
well as _train folder, in _train folder, the `input_gdisko_gauss_r10_s10` will contain only
frames corresponding ot the stylized keyframes, e.g., 001.png for Maruska640 sequence or 
000.png, 030.png, 070.png, and 103.png for Zuzka2 sequence. To train, do not forget to use
the correct config file, e.g., `--config "_config/reference_P_disco1010.yaml"` while running 
`train.py` script. To run the inference `generate.py` script, use an optional 
argument `--dir_x1 input_gdisko_gauss_r10_s10` that will tell the `generate.py` 
to load images from `input_gdisko_gauss_r10_s10`.

#### Masks for Gauss
While running the `gauss.exe`, the gaussian mixtures are generated for every mask image,
and are propagated to the sequence using optical flow, if there are multiple mask images 
provided, the resulting gaussian circles will be stacked on top of each other (and they will cover
potential holes). The mask can (and in most cases will) be fully-white 
images. If you are not sure what frames to pick as mask, pick the same as your keyframes 
or/and first and last frame of the sequence. See the gaussian mixture results, e.g., 
`input_gdisko_gauss_r10_s10`, if there are large black holes (larger than 100x100 px),
add one more mask image for the frame where the black holes are the largest.


### Build Temporal Consistency Tools

#### Build disflow
On Windows, try to use prebuilt `disflow.exe`. Otherwise, use `_tools/disflow/build_win.bat` 
to build `disflow.exe` yourself (on Linux/MacOS, get inspired by 
the build script, it should be really easy to build it). As it links against OpenCV-4.2.0,
it expects the `opencv_world420.dll` in PATH. Download [OpenCV-4.2.0](https://opencv.org/opencv-4-2-0/),
they offer prebuilt 
[Win pack](https://sourceforge.net/projects/opencvlibrary/files/4.2.0/opencv-4.2.0-vc14_vc15.exe/download). 
Feel free to modify the build script to use a different version of OpenCV. Note, OpenCV includes are 
provided and located at `_tools\disflow\opencv-4.2.0\include`, Windows .lib files are provided and 
located at `_tools\disflow\opencv-4.2.0\lib`.

#### Build bilateralAdv
On Windows, try to use prebuilt `bilateralAdv.exe`. Otherwise, use `_tools/bilateralAdv/build_win.bat` 
to build `bilateralAdv.exe` yourself (on Linux/MacOS, get inspired by 
the build script, it should be really easy to build it).

#### Build gauss
On Windows, try to use prebuilt `gauss.exe`. Otherwise, use `_tools/gauss/build_win.bat` 
to build `gauss.exe` yourself (on Linux/MacOS, get inspired by 
the build script, it should be really easy to build it).



## Repo TODO List
* Add code for "interactive" use-case as shown in the paper.
* Remove the dependency on tensorflow.


## Credits
* This project started when [Ondrej Texler](https://ondrejtexler.github.io/) was an 
  intern at [Snap Inc.](https://www.snap.com/), and it was funded 
  by [Snap Inc.](https://www.snap.com/) 
  and [Czech Technical University in Prague](https://www.cvut.cz/en)
* The main engineering forces behind this repository 
  are [Ondrej Texler](https://ondrejtexler.github.io/), 
  [David Futschik](https://dcgi.fel.cvut.cz/people/futscdav), and 
  [Michal Kučera](https://www.linkedin.com/in/kuceram/).
* The main engineering forces behind temporal consistency tools 
  are [Ondrej Jamriska](https://dcgi.fel.cvut.cz/people/jamriond) 
  and [Sarka Sochorova](https://dcgi.fel.cvut.cz/people/sochosar)


## License
* The Patch-Based Training method is not patented, and we do not plan on patenting. 
* However, you should be aware that certain parts of the code in this repository 
  were written when [Ondrej Texler](https://ondrejtexler.github.io/) 
  and [David Futschik](https://dcgi.fel.cvut.cz/people/futscdav) were employed 
  by [Snap Inc.](https://www.snap.com/). If you find this project useful for your 
  commercial interests, please, reimplement it.


## <a name="CitingFewShotPatchBasedTraining"></a>Citing
If you find Interactive Video Stylization Using Few-Shot Patch-Based Training useful 
for your research or work, please use the following BibTeX entry.

```
@Article{Texler20-SIG,
    author    = "Ond\v{r}ej Texler and David Futschik and Michal Ku\v{c}era and Ond\v{r}ej Jamri\v{s}ka and \v{S}\'{a}rka Sochorov\'{a} and Menglei Chai and Sergey Tulyakov and Daniel S\'{y}kora",
    title     = "Interactive Video Stylization Using Few-Shot Patch-Based Training",
    journal   = "ACM Transactions on Graphics",
    volume    = "39",
    number    = "4",
    pages     = "73",
    year      = "2020",
}
```

