from PIL import Image
import os

for i in range(3):
    folders = ["/home/nikodem/ANN/scissors", "/home/nikodem/ANN/rock", "/home/nikodem/ANN/paper"]
    directory = folders[i]
    c=0
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            im = Image.open(directory + "/" + filename)
            name = str(c) + '.png'
            rgb_im = im.convert('RGB')
            name = directory + "/" + name
            os.remove(directory + "/" + filename) 
            rgb_im.save(name)
            file_list = '"' + name + '"'
            c+=1
            #print(os.path.join(directory, filename))
            print(file_list)
            continue
        else:
            continue
    print()