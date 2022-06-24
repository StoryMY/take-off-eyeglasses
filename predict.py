""" Cog for removing eyeglasses and shadow."""
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import argparse
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from cog import BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms

from easy_use import *
from models.domain_adaption import DomainAdapter
from models.networks import ResnetGenerator, ResnetGeneratorMask


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # custom args
        self.img_size = 512
        self.ckpt_path = "./ckpt/pretrained.pt"

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}......")
        print("Loading models......")
        # load models
        norm = nn.BatchNorm2d
        self.DA_Net = DomainAdapter().to(self.device)
        self.GlassMask_Net = ResnetGeneratorMask(
            input_nc=64, output_nc=2, norm_layer=norm
        ).to(
            self.device
        )  # shadow prediction (mask)
        self.ShadowMask_Net = ResnetGeneratorMask(
            input_nc=65, output_nc=2, norm_layer=norm
        ).to(
            self.device
        )  # shadow prediction (mask)
        self.DeShadow_Net = ResnetGenerator(
            input_nc=5, output_nc=3, norm_layer=norm
        ).to(self.device)
        self.DeGlass_Net = ResnetGenerator(input_nc=4, output_nc=3, norm_layer=norm).to(
            self.device
        )

        # load ckpt
        ckpt = torch.load(self.ckpt_path)
        self.DA_Net.load_state_dict(ckpt["DA"])
        self.DA_Net.eval()
        self.GlassMask_Net.load_state_dict(ckpt["GM"])
        self.GlassMask_Net.eval()
        self.ShadowMask_Net.load_state_dict(ckpt["SM"])
        self.ShadowMask_Net.eval()
        self.DeShadow_Net.load_state_dict(ckpt["DeShadow"])
        self.DeShadow_Net.eval()
        self.DeGlass_Net.load_state_dict(ckpt["DeGlass"])
        self.DeGlass_Net.eval()

        # transform
        self.transform = transforms.Compose(
            [
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""

        img = str(image)

        # forward
        with torch.no_grad():
            img = Image.open(img).convert("RGB")
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)

            gfm, sfm = self.DA_Net(img)
            gmask = out2mask(self.GlassMask_Net(gfm), False)
            smask = out2mask(self.ShadowMask_Net(torch.cat([sfm, gmask], dim=1)), True)

            ds_in = torch.cat([img, smask, gmask], dim=1)
            ds_out = self.DeShadow_Net(ds_in)
            ds_out_masked = ds_out * (1 - gmask)
            dg_in = torch.cat([ds_out_masked, gmask], dim=1)
            dg_out = self.DeGlass_Net(dg_in)

            # save output image as Cog Path object
            output_path = Path(tempfile.mkdtemp()) / "output.png"

            savetensor2img(dg_out, output_path)
            savetensor2img(dg_out, "outtt.png")

            print(output_path)
            return output_path
