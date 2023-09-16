from model.networks import Generator
from utils.tools import get_config
import os
import torch
myDir = os.path.dirname(__file__)
class DummyTrainer(torch.nn.Module):
    def __init__(self,config,cuda=True,device_ids='cuda:0'):
        super().__init__()
        self.netG = Generator(config['netG'], cuda, device_ids)
        pass
    pass
class CAInpainter(object):
    def __init__(self, batch_size, checkpoint_dir,cuda=True,device_ids='cuda:0'):
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
        
    def forward(self,image,mask):
        # https://github.com/daa233/generative-inpainting-pytorch/blob/c6cdaea0427b37b5b38a3f48d4355abf9566c659/test_tf_model.py#L30
        x = (image / 127.5 - 1) * (1 - mask).cuda()
        with torch.no_grad():
            _, result, _ = self.netG(x, mask)  
        import ipdb;ipdb.set_trace()
        