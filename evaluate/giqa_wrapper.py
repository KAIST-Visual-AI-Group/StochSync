import subprocess 
import glob

save_dir = "/home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/results/panorama/ood/eval/metric_dict_2"

fdir1 = "/home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/results/GT_panorama/ood/perspective_2"
N_REF = len(list(glob.glob(f"{fdir1}/*")))

fdir2_list = []
method_list = []
ROOT_FDIR2 = "/home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/results/panorama/ood/eval/pers"
for fdir2 in glob.glob(f"{ROOT_FDIR2}/*"):
    fdir2_list.append(fdir2)
    method_list.append(fdir2.split("/")[-1])
    
# Add baselines
# ==============================================================
fdir2_list.append("/home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/results/baselines/ood/panfusion/pers")
method_list.append("panfusion")

fdir2_list.append("/home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/results/baselines/ood/mvdiffusion/pers")
method_list.append("mvdiffusion")
# ==============================================================

for i in range(len(fdir2_list)):
    fdir2 = fdir2_list[i]
    method = method_list[i]

    N_TGT = len(list(glob.glob(f"{fdir2}/*")))
    print(f"Total images : {N_REF}, {N_TGT}", method)

    assert N_REF == N_TGT
    assert fdir1 != fdir2

    command = f"python /home/jh27kim/code/current_projects/iclr2025/DistillAnywhere/scripts/eval/giqa_eval.py \
                --ref_image_dir {fdir1} \
                --fake_image_dir {fdir2} \
                --save_dir {save_dir} \
                --method {method}"

    subprocess.call(command, shell=True)
    
