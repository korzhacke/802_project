import glob
from shutil import copyfile

folders = glob.glob('./gan_all_images')
folders.sort()
i = 0
for folder in folders:
    cropped_images = glob.glob(folder + '/*.png')
    for cropped_image in cropped_images:
        if i < 3000:
            copyfile(cropped_image, 'train_all/gan_all/' + cropped_image.split('/')[2])
        if i > 3000 and i < 6000:
            copyfile(cropped_image, 'validation_all/gan_all/' + cropped_image.split('/')[2])
        i += 1


##for file in images:
##    for folder in folders:
##        idx = folder.split('/')[1].split('-')[0][:3]
##        if file[:3] == idx:
##            break
##    cropped_images = glob.glob(folder + '/*.png')
##    for cropped_image in cropped_images:
##        if file == cropped_image.split('/')[2]:
##            copyfile(folder + '/' + file, 'gan_good_images/' + file)
