import os 
fp = open("custom_validation.txt","w+")
base_path = "dataset/visda-2017/validation"
class_name = os.listdir(base_path)
class_name.sort()
for idx, cls in enumerate(class_name):
    cls_path = os.path.join(base_path,cls)
    file_names=  os.listdir(cls_path)
    for f in file_names:
        full_path = os.path.join(cls_path,f) + " " + str(idx)
        print(full_path)
        fp.write(full_path)
        fp.write("\n")
fp.close()


