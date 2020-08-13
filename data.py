import os
import torch.utils.data
import PIL
from PIL import Image
from torch.nn import functional as F
from custom_transforms import *


def get_geometric_blur_patch(tensor_small, midpoint, patchsize, coeff):
    midpoint = midpoint // coeff
    hs = patchsize // 2
    hn = max(0, midpoint[0] - hs)
    hx = min(midpoint[0] + hs, tensor_small.size()[1] - 1)
    xn = max(0, midpoint[1] - hs)
    xx = min(midpoint[1] + hs, tensor_small.size()[2] - 1)

    p = tensor_small[:, hn:hx, xn:xx]
    if p.size()[1] != patchsize or p.size()[2] != patchsize:
        r = torch.zeros((3, patchsize, patchsize))
        r[:, 0:p.size()[1], 0:p.size()[2]] = p
        p = r
    return p


################################
# Dataset full-images
################################
class DatasetFullImages(torch.utils.data.Dataset):
    def __init__(self, dir_pre, dir_post, dir_mask, device, dir_x1, dir_x2, dir_x3, dir_x4, dir_x5, dir_x6, dir_x7, dir_x8, dir_x9):
        super(DatasetFullImages, self).__init__()

        self.dir_pre = dir_pre
        self.fnames = sorted(os.listdir(self.dir_pre))
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.transform = build_transform()
        self.mask_transform = build_mask_transform()
        #self.temporal_frames = 3

        self.dir_pre_x1 = dir_x1
        self.dir_pre_x2 = dir_x2
        self.dir_pre_x3 = dir_x3
        self.dir_pre_x4 = dir_x4
        self.dir_pre_x5 = dir_x5
        self.dir_pre_x6 = dir_x6
        self.dir_pre_x7 = dir_x7
        self.dir_pre_x8 = dir_x8
        self.dir_pre_x9 = dir_x9
        #print('DatasetFullImages: number of training examples %d' % len(self.fnames))

    #def getitem_inner(self, item):
    def __getitem__(self, item):
        # get an image that is NOT stylized and its stylized counterpart
        fileName = self.fnames[item]
        pre = PIL.Image.open(os.path.join(self.dir_pre, fileName))
        pre_tensor = self.transform(pre)
        if self.dir_pre_x1 is not None and self.dir_pre_x1 != "":
            pre_x1 = PIL.Image.open(os.path.join(self.dir_pre_x1, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x1)), dim=0)
        if self.dir_pre_x2 is not None and self.dir_pre_x2 != "":
            pre_x2 = PIL.Image.open(os.path.join(self.dir_pre_x2, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x2)), dim=0)
        if self.dir_pre_x3 is not None and self.dir_pre_x3 != "":
            pre_x3 = PIL.Image.open(os.path.join(self.dir_pre_x3, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x3)), dim=0)
        if self.dir_pre_x4 is not None and self.dir_pre_x4 != "":
            pre_x4 = PIL.Image.open(os.path.join(self.dir_pre_x4, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x4)), dim=0)
        if self.dir_pre_x5 is not None and self.dir_pre_x5 != "":
            pre_x5 = PIL.Image.open(os.path.join(self.dir_pre_x5, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x5)), dim=0)
        if self.dir_pre_x6 is not None and self.dir_pre_x6 != "":
            pre_x6 = PIL.Image.open(os.path.join(self.dir_pre_x6, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x6)), dim=0)
        if self.dir_pre_x7 is not None and self.dir_pre_x7 != "":
            pre_x7 = PIL.Image.open(os.path.join(self.dir_pre_x7, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x7)), dim=0)
        if self.dir_pre_x8 is not None and self.dir_pre_x8 != "":
            pre_x8 = PIL.Image.open(os.path.join(self.dir_pre_x8, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x8)), dim=0)
        if self.dir_pre_x9 is not None and self.dir_pre_x9 != "":
            pre_x9 = PIL.Image.open(os.path.join(self.dir_pre_x9, fileName))
            pre_tensor = torch.cat((pre_tensor, self.transform(pre_x9)), dim=0)

        result = {'pre': pre_tensor,
                  'file_name': self.fnames[item]}

        if not self.dir_post.endswith("ignore"):
            post = PIL.Image.open(os.path.join(self.dir_post, fileName))
            post_tensor = self.transform(post)
            result['post'] = post_tensor

            # get a random already stylized image
            already_path = os.path.join(self.dir_post, self.fnames[np.random.randint(0, len(self.fnames))])
            im_s = PIL.Image.open(already_path)
            im_s_tensor = self.transform(im_s)
            result['already'] = im_s_tensor

        if not self.dir_mask.endswith("ignore"):
            mask = PIL.Image.open(os.path.join(self.dir_mask, fileName))
            mask = mask.point(lambda p: p > 128 and 255)  # !!! thresholding the mask fixes possible float and int conversion errors
            mask_tensor = self.mask_transform(mask).int().float()
            result['mask'] = mask_tensor

        return result

    def XXX__getitem__(self, item):
        result = {'pre': None,
                  'file_name': self.fnames[item]}

        for i in range(item - self.temporal_frames, item + self.temporal_frames + 1):
            is_curr_item = True if i == item else False
            i = max(0, i)
            i = min(len(self.fnames)-1, i)
            result_i = self.getitem_inner(i)

            if result['pre'] is None:
                result['pre'] = result_i['pre']
            else:
                result['pre'] = torch.cat((result['pre'], result_i['pre']), dim=0)

            if is_curr_item and "post" in result_i:
                result['post'] = result_i['post']
            if is_curr_item and "already" in result_i:
                result['already'] = result_i['already']
            if is_curr_item and "mask" in result_i:
                result['mask'] = result_i['mask']

        return result


    def __len__(self):
        return int(len(self.fnames))


#####
# Default "patch" dataset, used for training
#####
class DatasetPatches_M(torch.utils.data.Dataset):
    def __init__(self, dir_pre, dir_post, dir_mask, patch_size, device, dir_x1, dir_x2, dir_x3, dir_x4, dir_x5, dir_x6, dir_x7, dir_x8, dir_x9):
        super(DatasetPatches_M, self).__init__()
        self.dir_pre = dir_pre
        self.dir_post = dir_post
        self.dir_mask = dir_mask
        self.patch_size = patch_size

        self.geom_blur_coeff = 0.0
        self.device = "cpu"
        self.real_device = device
        #self.temporal_frames = 3

        self.paths_pre = sorted(os.listdir(dir_pre))
        self.paths_post = sorted(os.listdir(dir_post))
        self.paths_masks = sorted(os.listdir(dir_mask))

        self.transform = build_transform()
        self.mask_transform = build_mask_transform()

        self.images_pre = []
        self.images_pre_geom = []
        self.images_post = []
        images_mask = []

        # additional guides
        self.images_x1 = []
        self.images_x2 = []
        self.images_x3 = []
        self.images_x4 = []
        self.images_x5 = []
        self.images_x6 = []
        self.images_x7 = []
        self.images_x8 = []
        self.images_x9 = []

        i = 0
        for p in self.paths_pre:
            if p == "Thumbs.db":
                continue

            p_png = os.path.splitext(p)[0] + '.png'
            preim = PIL.Image.open(os.path.join(self.dir_pre, p))
            postim = PIL.Image.open(os.path.join(self.dir_post, p_png))
            maskim = PIL.Image.open(os.path.join(self.dir_mask, p_png))

            maskim = maskim.point(lambda p: p > 128 and 255)  # !!! thresholding the mask fixes possible float and int conversion errors

            pre_tensor = self.transform(preim)
            if self.geom_blur_coeff != 0.0:
                self.images_pre_geom.append(torch.nn.functional.interpolate(pre_tensor.unsqueeze(0), scale_factor=1.0 / self.geom_blur_coeff).squeeze(0))
            self.images_pre.append(pre_tensor)  # .to(self.device))

            if dir_x1 is not None and dir_x1 != "":
                x1_im = PIL.Image.open(os.path.join(dir_x1, p))
                self.images_x1.append(self.transform(x1_im))
            if dir_x2 is not None and dir_x2 != "":
                x2_im = PIL.Image.open(os.path.join(dir_x2, p))
                self.images_x2.append(self.transform(x2_im))
            if dir_x3 is not None and dir_x3 != "":
                x3_im = PIL.Image.open(os.path.join(dir_x3, p))
                self.images_x3.append(self.transform(x3_im))
            if dir_x4 is not None and dir_x4 != "":
                x4_im = PIL.Image.open(os.path.join(dir_x4, p))
                self.images_x4.append(self.transform(x4_im))
            if dir_x5 is not None and dir_x5 != "":
                x5_im = PIL.Image.open(os.path.join(dir_x5, p))
                self.images_x5.append(self.transform(x5_im))
            if dir_x6 is not None and dir_x6 != "":
                x6_im = PIL.Image.open(os.path.join(dir_x6, p))
                self.images_x6.append(self.transform(x6_im))
            if dir_x7 is not None and dir_x7 != "":
                x7_im = PIL.Image.open(os.path.join(dir_x7, p))
                self.images_x7.append(self.transform(x7_im))
            if dir_x8 is not None and dir_x8 != "":
                x8_im = PIL.Image.open(os.path.join(dir_x8, p))
                self.images_x8.append(self.transform(x8_im))
            if dir_x9 is not None and dir_x9 != "":
                x9_im = PIL.Image.open(os.path.join(dir_x9, p))
                self.images_x9.append(self.transform(x9_im))

            self.images_post.append(self.transform(postim))  # .to(self.device))
            images_mask.append(self.mask_transform(maskim).int().float().to(device))
            i += 1


        self.valid_indices = []
        self.valid_indices_left = []
        i = 0
        erosion_weights = torch.ones((1, 1, 7, 7)).to(device)
        for m in images_mask:
            m[m < 0.4] = 0
            m = F.conv2d(m.unsqueeze(0), erosion_weights, stride=1, padding=3)
            m[m < erosion_weights.numel()] = 0
            m /= erosion_weights.numel()

            self.valid_indices.append(m.squeeze().nonzero(as_tuple=False).to(self.device))
            self.valid_indices_left.append(list(range(0, len(self.valid_indices[i]))))
            i += 1


    def cut_patch(self, im, midpoint, size):
        hs = size // 2
        hn = max(0, midpoint[0] - hs)
        hx = min(midpoint[0] + hs, im.size()[1] - 1)
        xn = max(0, midpoint[1] - hs)
        xx = min(midpoint[1] + hs, im.size()[2] - 1)

        p = im[:, hn:hx, xn:xx]
        if p.size()[1] != size or p.size()[2] != size:
            r = torch.zeros((3, size, size))
            r[:, 0:p.size()[1], 0:p.size()[2]] = p
            p = r

        return p

    # CURRENTLY NOT IN USE
    def patch_diff(self, im, patch1_mid, patch2_mid, size):
        patch1 = self.cut_patch(im, patch1_mid, size)
        patch2 = self.cut_patch(im, patch2_mid, size)

        patch = patch1 - patch2
        patch = patch ** 2

        sum = patch.sum()

        return sum

    def cut_patches(self, im_index, midpoint, midpoint_r, size):
        patch_pre = self.cut_patch(self.images_pre[im_index], midpoint, size)
        if self.geom_blur_coeff != 0.0:
            geom_blur_patch = get_geometric_blur_patch(self.images_pre_geom[im_index], midpoint, size, self.geom_blur_coeff)
            patch_pre = torch.cat((patch_pre, geom_blur_patch), dim=0)

        if len(self.images_x1) > 0:
            patch_x1 = self.cut_patch(self.images_x1[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x1), dim=0)
        if len(self.images_x2) > 0:
            patch_x2 = self.cut_patch(self.images_x2[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x2), dim=0)
        if len(self.images_x3) > 0:
            patch_x3 = self.cut_patch(self.images_x3[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x3), dim=0)
        if len(self.images_x4) > 0:
            patch_x4 = self.cut_patch(self.images_x4[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x4), dim=0)
        if len(self.images_x5) > 0:
            patch_x5 = self.cut_patch(self.images_x5[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x5), dim=0)
        if len(self.images_x6) > 0:
            patch_x6 = self.cut_patch(self.images_x6[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x6), dim=0)
        if len(self.images_x7) > 0:
            patch_x7 = self.cut_patch(self.images_x7[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x7), dim=0)
        if len(self.images_x8) > 0:
            patch_x8 = self.cut_patch(self.images_x8[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x8), dim=0)
        if len(self.images_x9) > 0:
            patch_x9 = self.cut_patch(self.images_x9[im_index], midpoint, size)
            patch_pre = torch.cat((patch_pre, patch_x9), dim=0)

        patch_post = self.cut_patch(self.images_post[im_index], midpoint, size)
        patch_random = self.cut_patch(self.images_post[im_index], midpoint_r, size)

        return patch_pre, patch_post, patch_random

    def __getitem__(self, item):
        im_index = item % len(self.images_pre)
        midpoint_id = np.random.randint(0, len(self.valid_indices_left[im_index]))
        midpoint_r_id = np.random.randint(0, len(self.valid_indices[im_index]))
        midpoint = self.valid_indices[im_index][self.valid_indices_left[im_index][midpoint_id], :].squeeze()
        midpoint_r = self.valid_indices[im_index][midpoint_r_id, :].squeeze()

        del self.valid_indices_left[im_index][midpoint_id]
        if len(self.valid_indices_left[im_index]) < 1:
            self.valid_indices_left[im_index] = list(range(0, len(self.valid_indices[im_index])))

        result = {}

        for i in range(0, 1):  #range(im_index - self.temporal_frames, im_index + self.temporal_frames + 1):
            is_curr_item = True # if i == im_index else False
            #i = max(0, i)
            #i = min(len(self.images_pre)-1, i)

            patch_pre, patch_post, patch_random = self.cut_patches(im_index, midpoint, midpoint_r, self.patch_size)

            if "pre" not in result:
                result['pre'] = patch_pre
            else:
                result['pre'] = torch.cat((result['pre'], patch_pre), dim=0)

            if is_curr_item:
                result['post'] = patch_post
            if is_curr_item:
                result['already'] = patch_random

        return result

    def __len__(self):
        return sum([(n.numel() // 2) for n in self.valid_indices]) * 5  # dont need to restart


