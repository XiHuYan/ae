# configure config.py first

python train.py --exp_id v1.0 --datasets Deng --lr 1e-4 --eps 10 --bs 10 --gpu_id 0

python infer.py --exp_id v1.0 --datasets Deng --gpu_id 0

python visualize2D.py --exp_id v1.0 --datasets Deng
