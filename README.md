### VisionTransformer
An implementation of VisionTransformer for masked Spec self training in MAE Style

##### Requirements
Python 3.9

##### Installation
pip install -r requirements.txt


### Usage
##### Test trained models
There are already pretrained Models to use, which can be found here [google drive](https://drive.google.com/drive/folders/1axSJbtMP7gEV-eTJIXcHSsa-pmaewCnI?usp=sharing).
<br>The command for testing is: python3 train.py --modul_name=VisionTransformer --model_size=small --shuffle_type=os --test --checkpoint="Path To Model"
<br>For evaluation we use the veri_test2.txt, which can be found under the "files" folder.

##### Train your model
python3 train.py --modul_name=VisionTransformer  --model_size=small --shuffle_type=os --max_epochs=80
<br>The above configuration is enough to rerun our experiments.
<br>There are many more parameters available. For these inspect the train.py in the root. 

##### Prepare data
It is possible to use our voxceleb_prepare/voxceleb2_prepare in the "data" folder scripts to create the CSVs.
<br>Alternatively we deliever the CSVs already under "files/linux" where you can simple replace a part of the path

##### Code
The PytorchLithning Modul is direclty implemented in the train.py
<br>The PyTorch Modul for our Experiments can be found under "Modul.VisionTransformer"
<br>The Spectogramm processing can be found under "utils.features"
<br>In the "data" folder you find our "DataModul.py" and the "dataset.py" used for our experiments
<br>The Code used for generating our Plots can be found under "plotting" or "utils.Analysis"
<br>In "utils" there are many more files: for example calculating the cosine score, the amsoftmax loss, accuracy calculation, augmenting the data
