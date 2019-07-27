# cvwc2019_pose
Tiger Pose Estimation Using hrnet
For cvwc2019 pose estimation track
Based on https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

Please Follow the installation instruction from (deep-high-resolution-net ,above link).
i.e pytorch and the rest in requirements.txt

This repo contains additional dataloaders for the tiger dataset and 
experiment configuration file in experiments/tiger/hrnet/w32_288x384_adam_lr1e-3.yaml


Data is not here and must be downloaded from https://cvwc2019.github.io/challenge.html .
This must be put in data/tiger/pose/. It should have train, val and test folders.
including 'atrw_anno_pose_train' which contains the ground truth.

image_info_test.json contains image info for the test-set. This was generated
by gen_test_imageinfo.py.

On new test-set this should be generated again.
ex python gen_test_imageinfo.py <path-to-test-dir>
