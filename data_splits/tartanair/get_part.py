from pathlib import Path
import random

root_dir = Path("/date2/zhang_h/stereo/tartanair/tartanair_part")

subfolders = [item for item in root_dir.iterdir() if item.is_dir() and "soulcity" not in item.name]

all_samples = []

for i, subfolder in enumerate(subfolders):
    print(f"Processing subfolder {i + 1}/{len(subfolders)}")

    sample_path = subfolder / subfolder.name / "imgs"
    img_lft_dir = sample_path / "image_02" / "data"

    file_names = [file.name for file in img_lft_dir.iterdir() if file.is_file()]

    for file_name in file_names:
        idx = file_name[:-4].lstrip("0") or "0"
        line = f"{sample_path.relative_to(root_dir)} {idx} l"
        all_samples.append(line)

random.shuffle(all_samples)
num_train = int(0.9 * len(all_samples))


def write_txt(filename, sample_list):
    print(f"Writing {len(sample_list)} samples to {filename}")
    with open(filename, "w") as f:
        for line in sample_list:
            f.write(line + "\n")

write_txt(
    "/date2/zhang_h/stereo/TiO-Depth_pytorch/data_splits/tartanair/part.txt",
    all_samples,
)
write_txt(
    "/date2/zhang_h/stereo/TiO-Depth_pytorch/data_splits/tartanair/train_part.txt",
    all_samples[:num_train],
)
write_txt(
    "/date2/zhang_h/stereo/TiO-Depth_pytorch/data_splits/tartanair/test_part.txt",
    all_samples[num_train:],
)
