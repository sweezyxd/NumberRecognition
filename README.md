# NumberRecognition
Just a simple number recognition script I've made out of boredom today.

# Requirements
In order for the script to run successfully on your machine, you'll need to have the following packages installed:



-NumPy: `pip install numpy` (for linux: `pip3 install numpy`)

-Pandas: `pip install pandas` (for linux: `pip3 install pandas`)




Note: the script will probably not work on manchines with a python version lower than 3, concider updating ur python to 3.9 for better usage and overall experience.

# Usage
The script allows the user to train, save and load training models, and the usage is pretty simple.

First of all, you'll need to run the script:


`python main.py` (for linux: `python3 main.py`)



After that you can try the following commands:


`/start testing` ==> starts testing the model by giving it numbers indexes, and it'll give predictions.

`/start training` ==> starts training the model.

`/stop training` ==> stops training the model.

`/save "Model_name"` ==> saves the model with the given name.

`/load "Model_name"` ==> loads the model with the given name.

`/accuracy`  ==> shows the models accuracy (should be used only while training).

