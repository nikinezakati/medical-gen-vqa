from os import remove
from zipfile import ZipFile
import wget
import sys
from src.constants import RAW_DATA_DIR

def gather_raw_data(raw_data_output_dir=RAW_DATA_DIR):
    url = 'https://drive.google.com/u/0/uc?id=1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U&export=download&confirm=t&uuid=6133a381-1230-458b-8b3a-b22e17246879&at=AKKF8vwzp52Ov43TEltP9M5Ebdo5:1683556018455'
    wget.download(url,out = raw_data_output_dir)

    with ZipFile(raw_data_output_dir+'/Slake.zip', 'r') as zip_ref:
        zip_ref.extractall(raw_data_output_dir)
    remove(raw_data_output_dir+'/Slake.zip')



if __name__ == "__main__":
    try:
        dir = sys.argv[1]
    except:
        dir = RAW_DATA_DIR
    gather_raw_data(dir)