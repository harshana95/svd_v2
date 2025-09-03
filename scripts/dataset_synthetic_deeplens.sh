cd ..

FOLDER=/scratch/gilbreth/wweligam/dataset/deeplens_color_synthetic
# python 1.generate_psfs_synthetic.py --psf_type deeplens1 --save_dir $FOLDER
# python 2.generate_basis_psfs.py --psf_data_path $FOLDER
accelerate launch 3.generate_dataset_synthetic.py --opt ./checkpoints/dataset_svb.yml --psf_data_path $FOLDER --comment color

# FOLDER=/scratch/gilbreth/wweligam/dataset/deeplens_mono_synthetic
# python 1.generate_psfs_synthetic.py --psf_type deeplens2 --save_dir $FOLDER
# python 2.generate_basis_psfs.py --psf_data_path $FOLDER
# accelerate launch 3.generate_dataset_synthetic.py --opt ./checkpoints/dataset_svb.yml --psf_data_path $FOLDER --comment mono