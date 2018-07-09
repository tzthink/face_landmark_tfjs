feature_map = {
    'blur': {
        'clear': 0,
        'normal blur': 1,
        'heavy blur': 2,
    },
    'expression': {
        'typical expression': 0,
        'exaggerate expression': 1,
    },
    'illumination': {
        'normal illumination': 0,
        'extreme illumination': 1,
    },
    'occlusion': {
        'no occlusion': 0,
        'partial occlusion': 1,
        'heavy occlusion': 2,
    },
    'pose': {
        'typical pose': 0,
        'atypical pose': 1,
    },
    'invalid': {
        'false': 0,
        'true': 1,
    }
}

header = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']


def read_anno(anno_path):
    f_anno = open(anno_path)
    anno_dict = {}
    lines = f_anno.readlines()
    idx = 0
    while idx < len(lines):
        img_dict = {}
        img_name = lines[idx].strip('\n').strip()
        img_dict['filename'] = img_name
        idx += 1
        face_num = int(lines[idx].strip('\n').strip())
        img_dict['face_num'] = face_num
        idx += 1
        objects = []
        for i in range(face_num):
            face = lines[idx].strip('\n').strip()
            idx += 1
            face_feature = face.split(' ')
            obj = {}
            obj['x1'] = face_feature[0]
            obj['y1'] = face_feature[1]
            obj['w'] = face_feature[2]
            obj['h'] = face_feature[3]
            obj['blur'] = face_feature[4]
            obj['expression'] = face_feature[5]
            obj['illumination'] = face_feature[6]
            obj['invalid'] = face_feature[7]
            obj['occlusion'] = face_feature[8]
            obj['pose'] = face_feature[9]
            objects.append(obj)

        img_dict['object'] = objects
        anno_dict[img_name] = img_dict
    f_anno.close()
    return anno_dict


def main():
    train_anno_path = './wider_face_split/wider_face_train_bbx_gt.txt'
    val_anno_path = './wider_face_split/wider_face_val_bbx_gt.txt'
    train_anno_dict = read_anno(train_anno_path)
    val_anno_dict = read_anno(val_anno_path)


if __name__ == '__main__':
    main()