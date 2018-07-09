"""
Usage: python process_img_data.py

"""
import os.path


def create_examples_files(anno_path, ex_type):
    """
    Create train_examples files
    Create val_examples files
    """
    ex_f_name = './examples_files/' + ex_type + '.txt'
    if os.path.isfile(ex_f_name):
        return

    ex_f = open(ex_f_name, 'a+')
    with open(anno_path) as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            img_name = lines[idx]
            ex_f.write(img_name)
            idx += 1
            face_num = int(lines[idx].strip('\n').strip())
            idx += 1
            idx += face_num
    ex_f.close()


def main():
    """
    main function to process wider input data
    """
    train_anno_path = './wider_face_split/wider_face_train_bbx_gt.txt'
    val_anno_path = './wider_face_split/wider_face_val_bbx_gt.txt'
    create_examples_files(train_anno_path, 'train_examples')
    create_examples_files(val_anno_path, 'val_examples')


if __name__ == '__main__':
    main()
