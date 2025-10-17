python scripts/dfc/dfc_compute.py --dataset-name ines --wmin 5 --wmax 100 --wstep 1 --lag 1 --tau 3 --jobs 8 --cache overwrite --format 2D
python scripts/speed/dfc_speed_compute.py --dataset-name ines --window-min 5 --window-max 100 --window-step 1 --lag 1 --tau-max 3 --jobs 8 --cache overwrite
python scripts/bootstrap/compute_speed_bootstrap.py --dataset-name ines --subset all --group-cols Genotype,Sexe --pairs '('wt','F')-('wt','M');('dKI','F')-('dKI','M')' --tau-index 3 --pool-threshold median --n-boot 2000 --jobs 8 --progress
