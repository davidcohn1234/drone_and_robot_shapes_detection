# Drone and robot shapes detection
This project uses tello drone and robomaster ep for detecting 8 two-dimensional shapes.  
The 8 shapes are:  
circle, octagon, pentagon, rectangle, square, rhombus, star, triangle.  
I used `python 3.8.10`  
### Example output for robot:  
<img src="./resources_for_readme/gif_videos/robot/robomaster_ep_pov.gif" width="480" height="360" />

### Example output for tello drone:  
<img src="./resources_for_readme/gif_videos/drone/shapes_colors_on_screen.gif" width="480" height="360" />

## prerequisites
1. `Python 3.8` installed (python 3.10 will not with the robomaster sdk).
1. `tesseract` installed.
  - For installing `tesseract` in **Linux** follow this link: https://lindevs.com/install-tesseract-ocr-on-ubuntu
  - For installing `tesseract` in **Windows** follow this link: https://github.com/UB-Mannheim/tesseract/wiki
    - It is preferred that `tesseract.exe` will be in this path `C:/Program Files/Tesseract-OCR/tesseract.exe`.  In the code I first check this path and if it does not exist then I search for `tesseract.exe` in all the drives in the computer (which might take much longer).

## Instructions
For running the code to the following:
1. Open any folder you want to clone the project into.
1. right click on an empty area of the folder and click on `Open in Terminal`.
1. In the terminal, write the command `git clone git@github.com:davidcohn1234/drone_and_robot_shapes_detection.git` and wait for the command to finish clonning.
1. Open `Pycharm` then click on `File->Open` and choose the folder `drone_and_robot_shapes_detection` that you just clonned.
1. Click on `File->Settings...`. 
1. In the Settings windows click on `Project->Python Interpreter` then click on `Add Interpreter->Add Local Interpreter`.
1. in the window `Add Python Interpreter` make sure you're making a **new** environment and not an existing environemt and that you use base interpreter `python 3.8` (I don't think it will work for python 3.10). Then click on `OK` in all windows.
1. Open terminal in pycharm (you will see bellow the tab `Terminal`. Click on it).
1. Make sure you see `(venv)` at the beginning of the prompt in the terminal. If you don't see it then open a new terminal (the plus sign near the `Local` tab). In the new tab you're supposed to see the `(venv)`.
1. Write the command `pip install -r requirements.txt`
1. Click on the script name above and choose the script you want to run (can be either `script_drone_shapes_detection` or `script_robot_shapes_detection`.
1. Make sure the script is run on the environment you just created (In python Interpreter make sure the python path is in the environment you created).
1. Run the code

### Changing script parameters
1. You can choose parameters for each script (for the drone or for the robot)
1. For help for the robot script type
```bash
cd robot
python script_robot_shapes_detection.py --help
```
1. For help for the drone script type
```bash
cd drone
python script_drone_shapes_detection.py --help
```
1. For the drone script you can choose on which input folder to work on. Just go the script configuration and change the folder name in the `Parameters` line.  

1. For the robot script you can choose if you want to detect shapes by `contours` or by `yolo`. And you can also choose the input folder to work on.  
For that, go the script configuration and in the parameters line change the first word to the input folder name and the second word to either `DETECT_SHAPE_USING_CONTOURS` or `DETECT_SHAPE_USING_YOLO`.

