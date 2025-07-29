# Compile :  
pyinstaller trainModel.py --onefile --collect-all torch --debug noarchive   --consol
 # Run:
 trainModel.exe --task segment --model path\model.pt --data path\dataSet_MovingProduct_seg.yaml --epochs 100 --name test --project .\testFolder  
 
 or  
 
 trainModel.exe --task segment --model "path\model.pt" --data "path\dataSet_MovingProduct_seg.yaml" --epochs 100 --name test --project .\testFolder
 
 

 # Packages:
 main:
 ultralytics               8.3.44
 torch                     2.6.0+cu124
 pyinstaller               6.14.2

 Deatils: 
 Package                   Version
------------------------- ------------
altgraph                  0.17.4      
certifi                   2025.7.14   
charset-normalizer        3.4.2       
colorama                  0.4.6       
contourpy                 1.3.2       
cycler                    0.12.1      
filelock                  3.13.1      
fonttools                 4.59.0
fsspec                    2024.6.1
idna                      3.10
Jinja2                    3.1.4
kiwisolver                1.4.8
lap                       0.5.12
MarkupSafe                2.1.5
matplotlib                3.10.3
mpmath                    1.3.0
networkx                  3.3
numpy                     2.1.2
opencv-python             4.12.0.88
packaging                 25.0
pandas                    2.3.1
pefile                    2023.2.7
pillow                    11.0.0
pip                       25.1.1
psutil                    7.0.0
py-cpuinfo                9.0.0
pyinstaller               6.14.2
pyinstaller-hooks-contrib 2025.8
pyparsing                 3.2.3
python-dateutil           2.9.0.post0
pytz                      2025.2
pywin32-ctypes            0.2.3
PyYAML                    6.0.2
requests                  2.32.4
scipy                     1.15.3
seaborn                   0.13.2
setuptools                80.9.0
six                       1.17.0
sympy                     1.13.1
torch                     2.6.0+cu124
torchaudio                2.6.0+cu124
torchvision               0.21.0+cu124
tqdm                      4.67.1
typing_extensions         4.12.2
tzdata                    2025.2
ultralytics               8.3.44
ultralytics-thop          2.0.14
urllib3                   2.5.0
wheel                     0.46.1

 
