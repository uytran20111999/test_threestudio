import os

dirname = "./outputs"
all_exp = [f.path for f in os.scandir(dirname) if f.is_dir()]
# breakpoint()
ls = ""
use_gpu=3
for method in all_exp:
    all_prompt = [f.path for f in os.scandir(method) if f.is_dir()]
    for prompt in all_prompt:
        exp_path = f"{prompt}"
        parsed_yaml_path = f"{exp_path}/configs/parsed.yaml"
        last_ckpt_path = f"{exp_path}/ckpts/last.ckpt"
        ls+=f"python launch.py --config \"{parsed_yaml_path}\" --export --gpu {use_gpu} resume=\"{last_ckpt_path}\" system.exporter.save_uv=false system.exporter.save_texture=false system.exporter_type=mesh-exporter\n\n"

with open("./scripts/export_mesh2.sh","w") as f:
    f.write(ls)