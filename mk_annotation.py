import os

if __name__ == "__main__":

    datasets_path   = "datasets"

    types_name      = os.listdir(datasets_path)
    types_name      = sorted(types_name)

    list_file = open('annotation.txt', 'w')
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)

        for photo_name in photos_name:
            list_file.write('%s'%(os.path.join(os.path.abspath(datasets_path), type_name, photo_name)) + '\t' + str(cls_id))
            list_file.write('\n')
    list_file.close()
