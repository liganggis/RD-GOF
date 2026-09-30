python train.py -s ../../data/360_v2/bicycle/ -m exps/360/bicycle -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/bicycle -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/bicycle -r 4 --eval --cfg_path cfg/Mip360/360.yaml -BS 3

python train.py -s ../../data/360_v2/garden/ -m exps/360/garden -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/garden -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/garden -r 4 --eval --cfg_path cfg/Mip360/360.yaml -BS 4

python train.py -s ../../data/360_v2/stump/ -m exps/360/stump -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/stump -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/stump -r 4 --eval --cfg_path cfg/Mip360/360.yaml -BS 3

python train.py -s ../../data/360_v2/flowers/ -m exps//360/flowers -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/flowers -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/flowers -r 4 --eval --cfg_path cfg/Mip360/360.yaml -BS 2

python train.py -s ../../data/360_v2/treehill/ -m exps/360/treehill -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/treehill -r 4 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/treehill -r 4 --eval --cfg_path cfg/Mip360/360.yaml -BS 3

python train.py -s ../../data/360_v2/bonsai/ -m exps/360/bonsai -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/bonsai -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/bonsai -r 2 --eval --cfg_path cfg/Mip360/360.yaml -BS 10

python train.py -s ../../data/360_v2/counter/ -m exps/360/counter -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/counter -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/counter -r 2 --eval --cfg_path cfg/Mip360/360.yaml -BS 10

python train.py -s ../../data/360_v2/kitchen/ -m exps/360/kitchen -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/kitchen -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/kitchen -r 2 --eval --cfg_path cfg/Mip360/360.yaml -BS 10

python train.py -s ../../data/360_v2/room/ -m exps/360/room -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iterations 20000
python metrics2.py -m exps/360/room -r 2 --eval --cfg_path cfg/Mip360/360.yaml --iteration 20000
python extract_360_mesh.py -m exps/360/room -r 2 --eval --cfg_path cfg/Mip360/360.yaml -BS 10