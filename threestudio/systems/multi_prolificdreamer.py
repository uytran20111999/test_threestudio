import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import torch.nn as nn
from .prolificdreamer import ProlificDreamer
import random
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import copy
import glob
import re

def sort_key(file_path):
    # Extract the index from the file path using regular expressions
    match = re.search(r'(\d+)\.pt$', file_path)
    if match:
        index = int(match.group(1))
        return index
    else:
        # Return a large number for file paths that don't match the pattern
        return float('inf')

@threestudio.register("prolificdreamer_multi-system")
class ProlificDreamerMulti(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False
        system_type: str = "prolificdreamer_multi-system"
        n_particles: int = 4
        hiper_path: str = "/home/ubuntu/stable_dreamfusion/HiPer/all_ckpt"


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        

        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.prompt_utils = self.prompt_processor()
        self.n_particles = self.cfg.n_particles
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance).to(self.device)

        # omega_dict_settings = [copy.deepcopy(self.cfg.prompt_processor) for _ in range(self.n_particles)]
        # for i in range(len(omega_dict_settings)):
        #     omega_dict_settings[i]['prompt'] = f"A high quality photo of an <ice_cream{i}>"

        # self.prompt_processor = [threestudio.find(self.cfg.prompt_processor_type)(
        #     omega_dict_settings[i]
        # ) for i in range(self.n_particles)]
        # breakpoint()

        #delete duplicate tokenizer/text encoder
        # for i in range(1, self.n_particles):
        #     self.prompt_processor[i].destroy_text_encoder()
        #     self.prompt_processor[i].tokenizer = self.prompt_processor[0].tokenizer
        #     self.prompt_processor[i].text_encoder = self.prompt_processor[0].text_encoder
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_utils = self.prompt_processor()
        self.hiper_ckpt_path = sorted(glob.glob(f'{self.cfg.hiper_path}/*.pt'), key=sort_key)
        self.hiper_ckpt = [torch.load(self.hiper_ckpt_path[i]) for i in range(len(self.hiper_ckpt_path))]
        [p.requires_grad_(False) for p in self.hiper_ckpt]
        # breakpoint()
#         with torch.no_grad():
#             self.ref_images = self.guidance.sample(self.prompt_utils, elevation = torch.zeros(self.n_particles, device = self.device), 
#                                 azimuth = torch.zeros(self.n_particles, device = self.device) + 45.0, 
#                                 camera_distances = torch.zeros(self.n_particles, device = self.device) + 1.25, seed=self.global_step)

# #             img_save = torchvision.utils.make_grid(self.ref_images.permute(0,3,1,2)).permute(1,2,0).detach().cpu().numpy()
# #             plt.axis('off')
# #             plt.imshow(img_save)
# #             plt.savefig('foo.png')
#             transform = T.ToPILImage()
#             for i in range(self.ref_images.shape[0]):
# #                 breakpoint()
#                 img = transform(self.ref_images[i].permute(2,0,1))
#                 img.save(f"/home/ubuntu/stable_dreamfusion/threestudio/text_inverse/img_{i}.png")
#             breakpoint()
                

        # self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits8', pretrained=False)
        # self.dino.load_state_dict(torch.load("/home/ubuntu/stable_dreamfusion/threestudio/dino_deitsmall8_pretrain.pth"))
        # self.dino = self.dino.eval()
        # config_copy = self.cfg.copy()
        # del config_copy.n_particles
        del self.geometry
        del self.background

    def init_n_prolific(self, compose_nerf):
        # breakpoint()
        self.nerf_0 = compose_nerf[0]
        self.nerf_1 = compose_nerf[1]
        self.nerf_2 = compose_nerf[2]
        self.nerf_3 = compose_nerf[3]
        self.nerf_4 = compose_nerf[4]
        self.nerf_5 = compose_nerf[5]

    def forward(self, batch, idx) -> Dict[str, Any]:
        req_module = getattr(self,f"nerf_{idx}")
        return req_module.renderer(**batch)


    def on_fit_start(self) -> None:
        super().on_fit_start()

    def transform(self, x):
        x_224 = F.interpolate(x, (224,224))
        x_final = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(x_224)
        return x_final


    def training_step(self, batch, batch_idx):
        # breakpoint()
        idx = random.randint(0, self.n_particles -1)
        # idx1, idx2 = sample
        # idx = random.randint(0, self.n_particles -1)
        out = self(batch, idx)
        # selected_img = self.ref_images[idx1:idx1+1]
        # breakpoint()
        # selected_img = self.transform(selected_img.permute(0,3,1,2))
        # rendered_rgb = self.transform(out["comp_rgb"].permute(0,3,1,2))
        self.hiper_ckpt[idx] = self.hiper_ckpt[idx].to(device = out["comp_rgb"].device)
        # breakpoint()
        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, hiper_guidance = self.hiper_ckpt[idx]
            )
        else:
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False, hiper_guidance = self.hiper_ckpt[idx]
            )

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        # dino_rendered = self.dino(rendered_rgb)
        # dino_selected = self.dino(selected_img)
        # loss_MSE = F.mse_loss(dino_rendered, dino_selected)
        # with torch.no_grad():
        #     out_next = self(batch, idx2)
        #     rendered_rgb2 = self.transform(out_next["comp_rgb"].permute(0,3,1,2))
        #     dino_rendered2 = self.dino(rendered_rgb2)
        # loss_between_nerf = F.mse_loss(dino_rendered, dino_rendered2)
        # # loss += 0.05*loss_MSE
        # loss -= 0.1*loss_between_nerf
        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    out["weights"].detach()
                    * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (out["opacity"] > 0).sum()
                self.log("train/loss_orient", loss_orient)
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if "z_variance" in out:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            # sdf loss
            if "sdf_grad" in out:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
                self.log("train/inv_std", out["inv_std"], prog_bar=True)

        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        for i in range(self.n_particles):
            out = self(batch,i)
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-particle-{i}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="validation_step",
                step=self.true_global_step,
            )

            if self.cfg.visualize_samples:
                self.save_image_grid(
                    f"it{self.true_global_step}-{batch['index'][0]}-particle-{i}-sample.png",
                    [
                        {
                            "type": "rgb",
                            "img": self.guidance.sample(
                                self.prompt_utils[i], **batch, seed=self.global_step
                            )[0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.guidance.sample_lora(self.prompt_utils[i], **batch)[0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ],
                    name="validation_step_samples",
                    step=self.true_global_step,
                )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        for i in range(self.n_particles):
            out = self(batch,i)
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}-particle-{i}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        for i in range(self.n_particles):
            self.save_img_sequence(
                f"it{self.true_global_step}-particle-{i}-test",
                f"it{self.true_global_step}-test",
                f"(\d+)-particle-{i}\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step,
            )
    
    # def configure_optimizers(self):
    #     breakpoint()
    #     optim = parse_optimizer(self.cfg.optimizer, self)
    #     param_dicts = [name,value in self.cfg.optimizers.items()]
    #     #optimim[i] is parram[i]
    #     ret = {
    #         "optimizer": optim,
    #     }

    #     return ret