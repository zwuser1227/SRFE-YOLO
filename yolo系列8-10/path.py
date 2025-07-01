from pathlib import Path
 
sub_dir='Ultralytics'
 
#Windows目录
# path = Path.home() / 'AppData' / 'Roaming' / sub_dir
 
#MACOS目录
#path = Path.home() / 'Library' / 'Application Support' / sub_dir
 
#LINUX目录
path = Path.home() / '.config' / sub_dir
 
print(path)