import os
from PIL import Image

def get_list_of_folders():
  path = '/Users/tompease/Documents/Coding/airbnb/data/images'
  return os.listdir(path)

def get_list_of_files_in_folder(folder):
  path = f'/Users/tompease/Documents/Coding/airbnb/data/images/{folder}'
  return os.listdir(path)

def resize_image(final_size, im):
  size = im.size
  ratio = float(final_size) / max(size)
  new_image_size = tuple([int(x*ratio) for x in size])
  im = im.resize(new_image_size, Image.LANCZOS)
  new_im = Image.new("RGB", (final_size, final_size))
  new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
  return new_im

if __name__ == '__main__':
  original_image_folder = '/Users/tompease/Documents/Coding/airbnb/data/images'
  processed_image_folder = '/Users/tompease/Documents/Coding/airbnb/data/processed_images'

  dirs = get_list_of_folders()
  final_size = 512
  for folder in dirs:
    files = get_list_of_files_in_folder(folder)
    for photo in files:
      # Try block needed as there is an empty folder in each of the directories which throws an error
      try:
        im = Image.open(f'{original_image_folder}/{folder}/{photo}')
        new_im = resize_image(final_size, im)
        if not os.path.exists(f'{processed_image_folder}/{folder}'):
          os.makedirs(f'{processed_image_folder}/{folder}')
        new_im.save(f'{processed_image_folder}/{folder}/{photo}')
      except:
        pass