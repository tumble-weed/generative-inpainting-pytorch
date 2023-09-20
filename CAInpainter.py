from model.networks import Generator
from utils.tools import get_config
import os
import torch
import numpy as np
import dutils
myDir = os.path.dirname(__file__)
class DummyTrainer(torch.nn.Module):
    def __init__(self,config,cuda=True,device_ids='cuda:0'):
        super().__init__()
        self.netG = Generator(config['netG'], cuda, device_ids)
        pass
    pass
class CAInpainter(torch.nn.Module):
    def __init__(self, batch_size, checkpoint_dir,cuda=True,device_ids='cuda:0'):
        super().__init__()
        self.config = get_config(f'{myDir}/configs/config.yaml')
        self.dummy_trainer = DummyTrainer(config=self.config,cuda=cuda,device_ids=device_ids)
        self.dummy_trainer.load_state_dict(torch.load(os.path.join(myDir,f"{myDir}/torch_model.pt")))
        self.netG = self.dummy_trainer.netG
        self.netG.eval()
        # import ipdb;ipdb.set_trace()
        # self.netG = Generator(self.config['netG'], cuda, device_ids)
        # self.checkpoint_path = os.path.join(myDir,'checkpoints',
        #                                            config['dataset_name'],
        #                                            config['mask_type'] + '_' + config['expname'])
        # last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
        # self.netG.load_state_dict(torch.load(os.path.join(myDir,f"{myDir}/torch_model.pt")))
        #=============================================================================
        # quantities related to CAInpainter and used by generative-attribution-methods
        #=============================================================================
        pth_mean = np.ones((1, 3, 1, 1), dtype='float32')
        pth_mean[0, :, 0, 0] = np.array([0.485, 0.456, 0.406])
        pth_std = np.ones((1, 3, 1, 1), dtype='float32')
        pth_std[0, :, 0, 0] = np.array([0.229, 0.224, 0.225])
        pth_mean = torch.tensor(pth_mean)
        pth_std = torch.tensor(pth_std)
        self.upsample = torch.nn.Upsample(size=(256, 256), mode='bilinear')
        self.downsample = torch.nn.Upsample(size=(224, 224), mode='bilinear')        
        self.modules = torch.nn.ModuleList([self.upsample,self.downsample,self.netG])
        self.register_buffer('pth_mean',pth_mean)
        self.register_buffer('pth_std',pth_std)
    def forward(self,image,mask):
        # https://github.com/daa233/generative-inpainting-pytorch/blob/c6cdaea0427b37b5b38a3f48d4355abf9566c659/test_tf_model.py#L30
        if False:
            x = (image / 127.5 - 1) * (1 - mask).cuda()
        else:
            assert image.min() >= -1.
            assert image.max() <= 1.
            assert mask.shape[-2:] == image.shape[-2:]
            x = image * (1-mask)

        with torch.no_grad():
            import ipdb;ipdb.set_trace()
            stage1_result, result, offset_flow = self.netG(x, mask)  
        
        # import ipdb;ipdb.set_trace()
        return result
    def generate_background(self, pytorch_image, pytorch_mask, batch_process=False):
        '''
        Use to generate whole blurry images with pytorch normalization.
        '''
        # assert 'bool' in str(mask.dtype).lower()
        mask = pytorch_mask.expand(pytorch_mask.shape[0], 3, 224, 224)
        mask = mask[:,:1]
        if False:
            mask = self.upsample((mask)).data  # .round()
            mask = mask.cpu().numpy()
            thresh = max(0.5, 0.5 * (np.max(mask) + np.min(mask)))
            mask = (mask < thresh).astype(float)

            # Make it into tensorflow input ordering, then resizing then normalization
            # Do 3 things:
            # - Move from NCHW to NHWC, and from RGB to BGR input
            # - Normalize to 0 - 255 with integer round up
            # - Resize the image size to be 256 x 256
            mask = np.moveaxis(mask, 1, -1)*255
            # mask = (1. - mask) * 255
        else:
            assert mask.max() <= 1.
            assert mask.min() >= 0.
            mask = self.upsample((mask))
            mask = (mask > 0.5).float()
        image = self.upsample((pytorch_image))
        image = (image * self.pth_std + self.pth_mean)
        if False:
            image = self.upsample((pytorch_image)).data.cpu().numpy()
            image = np.round((image * self.pth_std + self.pth_mean) * 255)
            image = np.moveaxis(image, 1, -1)
            image = image[:, :, :, ::-1]

        # print('there will be multiple masks and a single image, so see if you have to repeat the image')
        # import ipdb;ipdb.set_trace()
        image = image.repeat(mask.shape[0],1,1,1)
        if False:
            # t1 = time.time()
            if batch_process:
                image = np.stack((image[0, :], )*mask.shape[0], axis=0)

            input_image = np.concatenate([image, mask], axis=2)
            # print(time.time() - t1)

            # DEBUG
            # import cv2
            # cv2.imwrite('./test_input.jpg', input_image[0])
            '''
            returns pytorch image and mask
            '''
        #https://github.com/tumble-weed/generative-inpainting-pytorch/blob/050ec08eacde19290255eff37ea686063c782a4c/test_tf_model.py#L37
        # mask in 0-1
        # mask of area to be filled is white
        pth_img = self.forward((image*2 - 1),1-mask)
        
        assert pth_img.max() <= 1
        assert pth_img.min() >= -1
        pth_img = (pth_img + 1)/2.
        # import ipdb;ipdb.set_trace()
        pth_img = ((pth_img ) - self.pth_mean) / self.pth_std
        pth_img = self.downsample((pth_img)).data
        if False:
            # t1 = time.time()
            tf_images = self.sess.run(self.output, {self.images_ph: input_image})
            # print(time.time() - t1)
            # print('#'*25)

            # it's RGB back. So just change back to pytorch normalization
            pth_img = np.moveaxis(tf_images, 3, 1)
            pth_img = ((pth_img / 255.) - self.pth_mean) / self.pth_std

            pth_img = pytorch_image.new(pth_img)
            pth_img = self.downsample(Variable(pth_img)).data

        return pth_img, mask

        