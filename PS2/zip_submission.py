import argparse
import os
import shutil
import yaml

def copy_directory(src, dest):
  try:
    shutil.copytree(src, dest)
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
  except OSError as e:
    print('Directory not copied. Error: %s' % e)

parser = argparse.ArgumentParser()
parser.add_argument('--gt_username', required=True, type=str)
args = parser.parse_args()

shutil.rmtree('temp_submission', ignore_errors=True)
os.mkdir('temp_submission')

dir_list = yaml.load(open('.zip_dir_list.yml'), Loader=yaml.BaseLoader)['directories']
for dir_name in dir_list:
	copy_directory(dir_name, '/'.join(['temp_submission', dir_name]))
shutil.make_archive(args.gt_username, 'zip', 'temp_submission')
shutil.rmtree('temp_submission', ignore_errors=True)
