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


## Run

Download the [TESTING_DATA](https://drive.google.com/file/d/1EscSNFg4ILpB7dxr-zYw_UdOILLmDlRj/view?usp=sharing), and unzip. 
The _train folder is expected to be next to the _gen folder. 

To train the network, run the `train.py`. 
To generate the results, run `generate.py`. 
See example commands below:

```
train.py --config "_config/reference_P.yaml" 
		 --data_root "Maruska640_train" 
		 --log_interval 1000 
		 --log_folder logs_reference_P
```

Every 1000 (log_interval) epochs, `train.py` saves the current generator to logs_reference_P (log_folder), and it validates/runs the generator on _gen data - the result is saved in Maruska640_gen/res__P


```
generate.py --checkpoint "Maruska640_train/logs_reference_P/model_00010.pth" 
	    --data_root "Maruska_gen"
		--dir_input "input_filtered"
	    --outdir "Maruska_gen/res_00010" 
	    --device "cuda:0"
```


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
torch                  1.6.0
torchvision            0.7.0
```


## Credits
* This project started when [Ondrej Texler](https://ondrejtexler.github.io/) was an intern at [Snap Inc.](https://www.snap.com/), and it was funded by [Snap Inc.](https://www.snap.com/) and [Czech Technical University in Prague](https://www.cvut.cz/en) 


## License
* Released for research purposes only.
* © Snap Inc. 2020


## <a name="CitingFewShotPatchBasedTraining"></a>Citing
If you find Interactive Video Stylization Using Few-Shot Patch-Based Training useful for your research or work, please use the following BibTeX entry.

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

