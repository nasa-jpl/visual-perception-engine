import os
import torch
from model_architectures.depth_anything_v2.dpt import DepthAnythingV2
from model_architectures.dino_foundation_model import DinoFoundationModel

output_dir = "models/checkpoints/".format(**os.environ)


if __name__ == "__main__":
    weights = {
        "vits": "models/checkpoints/depth_anything_v2_vits.pth".format(**os.environ),
        "vitb": "models/checkpoints/depth_anything_v2_vitb.pth".format(**os.environ),
        "vitl": "models/checkpoints/depth_anything_v2_vitl.pth".format(**os.environ),
    }

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    for encoder in model_configs.keys():
        model = DepthAnythingV2(ignore_xformers=True, **model_configs[encoder])
        model.load_state_dict(torch.load(weights[encoder]))

        # save pretrained weights
        model_weights = model.pretrained.state_dict()
        torch.save(model_weights, f"{output_dir}dinov2_{encoder}14_from_dav2.pth")
        print(f"Model weights for {encoder} saved")

        m = DinoFoundationModel(encoder=encoder, ignore_xformers=True)
        m.load_state_dict(model_weights)
        m.eval()
        m = m.cuda()

        assert m(torch.randn(1, 3, 518, 518).cuda())

        # save depth head weights
        model_weights = model.depth_head.state_dict()
        torch.save(model_weights, f"{output_dir}depth_anything_v2_head_{encoder}_from_dav2.pth")
        print(f"Depth head weights for {encoder} saved")
