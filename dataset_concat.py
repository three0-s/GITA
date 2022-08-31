import torch
import torchvision
import glob
from PIL import Image
from tqdm import tqdm

front_files = glob.glob('/home/yewon/GITA/dataset/train/front/*.jpg')
left_files = glob.glob('/home/yewon/GITA/dataset/train/left/*.jpg')

front_id_fnames = []
for front in front_files:
    front_img_id = front.split('/')[-1].split('.')[0].replace('_intraoral_front', '')
    front_dict = dict(img_id=front_img_id, path=front)
    front_id_fnames.append(front_dict)

left_id_fnames = []
for left in left_files:
    left_img_id = left.split('/')[-1].split('.')[0].replace('_intraoral_left', '')
    left_dict = dict(img_id=left_img_id, path=left)
    left_id_fnames.append(left_dict)

pair_count = 0
for front in tqdm(front_id_fnames, total=len(front_id_fnames)):
    idx = -1
    for i, left in enumerate(left_id_fnames):
        if front['img_id']==left['img_id']:
            idx=i
            break
    if idx != -1:
        pair_count+=1
        front_img = torchvision.transforms.PILToTensor()(Image.open(front['path']))
        left_img = torchvision.transforms.PILToTensor()(Image.open(left['path']))
        img = torch.concat([front_img, left_img], axis=2)
        img = torchvision.transforms.ToPILImage()(img)
        fname = '/home/yewon/GITA/dataset/train/'+front['img_id']+'_intraoral.png'
        img.save(fname)
        left_id_fnames.pop(idx)

print("="*8 + " COMPLETED! "  + "="*8)
print(f"Total {pair_count} numbers of images")
    