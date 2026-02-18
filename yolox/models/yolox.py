#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from torchvision.utils import save_image
import os
import cv2

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):

        # TODO: gt debug done, let's insert warp-unwarp code later!
        # NOTE: remember to change unwarp to resize each feature maps!
        
        # # ---- DEBUG: print input shape once ----
        # if targets is not None:
        #     print("[YOLOX] targets shape: ", tuple(targets.shape)) # (1, 3, 5)
        #     print("[YOLOX] targets full tensor:\n", targets)
        # print(f"[YOLOX] input x shape: {tuple(x.shape)}") # (1, 3, 800, 1440)


        fpn_outs = self.backbone(x)


        # # fpn output content features of [dark3, dark4, dark5]
        # print("[YOLOX] FPN shapes:", [tuple(f.shape) for f in fpn_outs]) # [(1, 320, 100, 180), (1, 640, 50, 90), (1, 1280, 25, 45)]
        # os.makedirs("debugs", exist_ok=True)
        # save_image(x, "debugs/image.png", normalize=True) # save the input image for visualization
        # _vis_bbox_tensor(x, targets, "debugs/image_gt.png")
        # save_image(fpn_outs[0][0:1,0:1], "debugs/fpn_out_0.png", normalize=True) # save the first channel of the first FPN output for visualization
        # save_image(fpn_outs[1][0:1,0:1], "debugs/fpn_out_1.png", normalize=True)
        # save_image(fpn_outs[2][0:1,0:1], "debugs/fpn_out_2.png", normalize=True)
        # print("[YOLOX] Saved input and FPN outputs as images in debugs/ folder for visualization.")
        # exit()


        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs




def _vis_bbox_tensor(x, targets, save_path):
    import cv2, numpy as np, torch, os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # take first image in batch
    img = x[0].detach().cpu()
    img = img.permute(1, 2, 0).contiguous().numpy()

    # simple de-normalize (YOLOX val uses mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)
    
    # --- ERROR WAS HERE ---
    # The [:, :, ::-1] makes the array non-contiguous. 
    # Adding .copy() fixes the memory layout.
    img = (img * 255).astype(np.uint8)[:, :, ::-1].copy() 

    if targets is None:
        cv2.imwrite(save_path, img)
        return

    t = targets[0].detach().cpu() if targets.ndim == 3 else targets.detach().cpu()

    for row in t:
        cls, cx, cy, w, h = row.tolist()
        if w <= 0 or h <= 0:
            continue
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        
        # Now this will work because img is contiguous
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imwrite(save_path, img)