import os

# python train.py -s ../../data/TNT/Barn -m ./exps/TNT/Barn_v1 -r2 --cfg_path cfg/TNT/Barn.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Barn_v1 -r 2 --cfg_path cfg/TNT/Barn.yaml --iteration 20000
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Barn --traj-path ../../data/TNT/Barn/Barn_COLMAP_SfM.log --ply-path ./exps/TNT/Barn_v1/fusion/mesh_0.ply
# python post_mesh.py

# ulimit -n 2048
# python train.py -s ../../data/TNT_by_gof/Courthouse -m ./exps/TNT/Courthouse_v1 -r2 --cfg_path cfg/TNT/Courthouse.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Courthouse_v1 -r 2 --cfg_path cfg/TNT/Courthouse.yaml --iteration 20000 -DT 2
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Courthouse --traj-path ../../data/TNT_by_gof/Courthouse/Courthouse_COLMAP_SfM.log --ply-path ./exps/TNT/Courthouse_v1/fusion/mesh_0.ply
# python post_mesh.py

# python train.py -s ../../data/TNT/Caterpillar -m ./exps/TNT/Caterpillar_v1 -r2 --cfg_path cfg/TNT/Caterpillar.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Caterpillar_v1 -r 2 --cfg_path cfg/TNT/Caterpillar.yaml --iteration 20000
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Caterpillar --traj-path ../../data/TNT/Caterpillar/Caterpillar_COLMAP_SfM.log --ply-path ./exps/TNT/Caterpillar_v1/fusion/mesh_0.ply
# python post_mesh.py

# python train.py -s ../../data/TNT/Ignatius -m ./exps/TNT/Ignatius_v1 -r2 --cfg_path cfg/TNT/Ignatius.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Ignatius_v1 -r 2 --cfg_path cfg/TNT/Ignatius.yaml --iteration 20000
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Ignatius --traj-path ../../data/TNT/Ignatius/Ignatius_COLMAP_SfM.log --ply-path ./exps/TNT/Ignatius_v1/fusion/mesh_0.ply
# python post_mesh.py

# python train.py -s ../../data/TNT/Meetingroom -m ./exps/TNT/Meetingroom_v1 -r2 --cfg_path cfg/TNT/Meetingroom.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Meetingroom_v1 -r 2 --cfg_path cfg/TNT/Meetingroom.yaml --iteration 20000
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Meetingroom --traj-path ../../data/TNT/Meetingroom/Meetingroom_COLMAP_SfM.log --ply-path ./exps/TNT/Meetingroom_v1/fusion/mesh_0.ply
# python post_mesh.py

# python train.py -s ../../data/TNT/Truck -m ./exps/TNT/Truck_v1 -r2 --cfg_path cfg/TNT/Truck.yaml --iterations 20000
# python extract_tnt_mesh2.py -m ./exps/TNT/Truck_v1 -r 2 --cfg_path cfg/TNT/Truck.yaml --iteration 20000 -DT 10
# python eval_tnt/run.py --dataset-dir ../../data/TNT_real/Truck --traj-path ../../data/TNT/Truck/Truck_COLMAP_SfM.log --ply-path ./exps/TNT/Truck_v1/fusion/mesh_0.ply
# python post_mesh.py
