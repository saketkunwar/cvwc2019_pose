import json
import glob
import sys
from skimage import io
import os


def main(test_dir):
    print (test_dir)
    g = glob.glob(test_dir + '/*.jpg')
    val = json.load(open('data/tiger/pose/atrw_anno_pose_train/keypoint_val.json'))

    print ('num images', len(g))
    tmp_json={"categories": val['categories'],'images' : [],'annotations':[],'type':'instances'}
    for i,item in enumerate(g):
        id  = int(item.split('/')[-1].replace('.jpg',''))
        img = io.imread(item)
        h,w,_ = img.shape
        tmp_json['images'].append({'filename':item.split('/')[-1],'height':h,'width':w, 'id': id})
        tmp_json['annotations'].append({'category_id' :1, 'keypoints':[0]*45,'num_keypoints': 0, 'image_id':id,'id':id,'area' :h*w, 'is_crowd':0 ,'bbox': [0, 0, w, h] })
    filename = os.path.join('data/tiger/pose/atrw_anno_pose_train/','image_info_test.json')
    json.dump(tmp_json, open(filename,'w'))

if __name__=='__main__':
    main(sys.argv[1])
