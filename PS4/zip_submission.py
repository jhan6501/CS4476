import os
import shutil
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_username', required=True, type=str)
args = parser.parse_args()

def copy_directory(src, dest):
  try:
    shutil.copytree(src, dest)
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
  except OSError as e:
    print('Directory not copied. Error: %s' % e)

shutil.rmtree('temp_submission', ignore_errors=True)
os.mkdir('temp_submission')
dir_list = yaml.load(open('.zip_dir_list.yml'), Loader=yaml.BaseLoader)['required_directories']

for dir_name in dir_list:
    copy_directory(dir_name, '/'.join(['temp_submission', dir_name]))

shutil.make_archive(args.gt_username + '_proj4', 'zip', 'temp_submission')
shutil.rmtree('temp_submission', ignore_errors=True)

out_file = args.gt_username + '_proj4' + '.zip'
if (os.path.getsize(out_file) > 50000000):
    os.remove(out_file)
    print("SUBMISSION DID NOT ZIP, ZIPPED SIZE > 50MB")
