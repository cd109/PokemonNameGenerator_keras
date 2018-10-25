import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config as cf

Dataset_Debug_Display = False


class DataLoader():

    def __init__(self, phase='Train', shuffle=False):
        self.ids = get_ids()
        self.voca = cf.Vocabrary
        self.datas = []
        self.last_mb = 0
        self.phase = phase
        self.gt_count = [0 for _ in range(1)]
        self.prepare_datas(shuffle=shuffle)
        
        
    def prepare_datas(self, shuffle=True):
        if self.phase == 'Train':
            dir_paths = cf.Train_dirs
        elif self.phase == 'Test':
            dir_paths = cf.Test_dirs
            
        print('------------\nData Load (phase: {})'.format(self.phase))
        
        for dir_path in dir_paths:
            files = []
            for ext in cf.File_extensions:
                files += glob.glob(dir_path + '/*{}'.format(ext))
            files.sort()
            load_count = 0
            
            for img_path in files:
                gt = self.get_gt(img_path)
                if cv2.imread(img_path) is None:
                    continue
                #if self.gt_count[gt] >= 10000:
                #    continue
                gt_path = 1
                
                data = {'img_path': img_path,
                        'gt_path': gt_path,
                        'h_flip': False,
                        'v_flip': False,
                        'rotate': False
                }
                
                self.datas.append(data)
                #self.gt_count[gt] += 1
                load_count += 1

            print(' - {} - {} datas -> loaded {}'.format(dir_path, len(files), load_count))

        #self.display_gt_statistic()
        if len(self.datas) == 0:
            raise Exception("data not found")

        if self.phase == 'Train':
            self.data_augmentation()            
            #self.display_gt_statistic()
        print(' Data num: {}'.format(len(self.datas)))
        self.set_index(shuffle=shuffle)

        
    def display_gt_statistic(self):
        print(' -*- Training label  -*-')
        print('   Total data: {}'.format(len(self.datas)))
        for i, gt in enumerate(self.gt_count):
            print('  - {} : {}'.format(cf.Class_label[i], gt))

    def set_index(self, shuffle=True):
        self.data_n = len(self.datas)
        self.indices = np.arange(self.data_n)
        if shuffle:
            np.random.seed(cf.Random_seed)
            np.random.shuffle(self.indices)

    def get_minibatch_index(self, shuffle=False):
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1
        _last = self.last_mb + mb
        if _last >= self.data_n:
            mb_inds = self.indices[self.last_mb:]
            self.last_mb = _last - self.data_n
            if shuffle:
                np.random.seed(cf.Random_seed)
                np.random.shuffle(self.indices)
            _mb_inds = self.indices[:self.last_mb]
            mb_inds = np.hstack((mb_inds, _mb_inds))
        else:
            mb_inds = self.indices[self.last_mb : self.last_mb+mb]
            self.last_mb += mb
        self.mb_inds = mb_inds

        
    def get_minibatch(self, shuffle=True):
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1

        self.get_minibatch_index(shuffle=shuffle)
        imgs = np.zeros((mb, cf.Height, cf.Width, 3), dtype=np.float32)
            
        decoder_inputs = np.zeros((mb, cf.Name_length, cf.Vocabrary_num), dtype=np.float32)
        decoder_outputs = np.zeros_like(decoder_inputs)
        
        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img = self.load_image(data['img_path'])
            img = self.image_dataAugment(img, data)
            gt = self.get_gt(data['img_path'])
            
            imgs[i] = img
            decoder_inputs[i] = gt
            decoder_outputs[i, 0:cf.Name_length-1] = gt[1:]
            
            if Dataset_Debug_Display:
                print(data['img_path'])
                print()
                plt.imshow(imgs[i])
                plt.subplots()
                plt.imshow(gts[i])
                plt.show()
        
        return imgs, decoder_inputs, decoder_outputs
        
    
    def get_gt(self, img_name):
        gt_vec = np.zeros((cf.Name_length, cf.Vocabrary_num), dtype=np.float32)
        fname = os.path.basename(img_name)
        fname, ext = os.path.splitext(fname)
        p_id, _ = fname.split('_')
        name = self.ids[p_id]
        gt_vec[0, self.voca.index(cf.EOS)] = 1.
        for i, char in enumerate(name):
            gt_vec[i+1, self.voca.index(char)] = 1.
        gt_vec[len(name)+1, self.voca.index(cf.EOS)] = 1.

        return gt_vec
        
    
    ## Below functions are for data augmentation
    def load_image(self, img_name):
        img = cv2.imread(img_name)
        
        if img is None:
            raise Exception('file not found: {}'.format(img_name))
        
        orig_h, orig_w = img.shape[:2]

        scaled_height = cf.Height
        scaled_width = cf.Width
        img = cv2.resize(img, (scaled_width, scaled_height))
        img = img[:, :, (2,1,0)]
        img = img / 255.

        return img


    def image_dataAugment(self, image, data):
        h, w = image.shape[:2]
        if data['h_flip']:
            image = image[:, ::-1]
        if data['v_flip']:
            image = image[::-1, :]
        if data['rotate']:
            max_side = max(h, w)
            if len(image.shape) == 3: 
                frame = np.zeros((max_side, max_side, 3), dtype=np.float32)
            elif len(image.shape) == 2:
                frame = np.zeros((max_side, max_side), dtype=np.float32)
            tx = int((max_side-w)/2)
            ty = int((max_side-h)/2)
            frame[ty:ty+h, tx:tx+w] = image
            M = cv2.getRotationMatrix2D((max_side/2, max_side/2), 90, 1)
            rot = cv2.warpAffine(frame, M, (max_side, max_side))
            image = rot[tx:tx+w, ty:ty+h]
            temp = h
            h = w
            w = temp
        return image
    
    
    def data_augmentation(self):
        print('   ||   -*- Data Augmentation -*-')
        if cf.Horizontal_flip:
            self.add_horizontal_flip()
            print('   ||    - Added horizontal flip')
        if cf.Vertical_flip:
            self.add_vertical_flip()
            print('   ||    - Added vertival flip')
        if cf.Rotate_ccw90:
            self.add_rotate_ccw90()
            print('   ||    - Added Rotate ccw90')
        print('  \  /')
        print('   \/')
        
    

    def add_horizontal_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['h_flip'] = True
            new_data.append(_data)
            #gt = self.get_gt(data['img_path'])
            #self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_vertical_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['v_flip'] = True
            new_data.append(_data)
            #gt = self.get_gt(data['img_path'])
            #self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_rotate_ccw90(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['rotate'] = True
            new_data.append(_data)
            #gt = self.get_gt(data['img_path'])
            #self.gt_count[gt] += 1
        self.datas.extend(new_data)


def get_ids():
    ids = {}
    with open(cf.gt_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            pokemon_id, name = line.split(' ')
            #pokemon_id = str(pokemon_id)
            ids[pokemon_id] = name

    return ids
