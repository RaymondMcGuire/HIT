import os

package_directory = os.path.dirname(os.path.abspath(__file__))
repo_directory = os.path.dirname(package_directory)

################## To edit, or set the matching environment variable
# Dataset (only needed for training and evaluation)
packaged_data_folder = os.environ.get('HIT_DATA_FOLDER', os.path.join(repo_directory, 'hit_dataset_v1.0', 'repackaged'))

# Trained HIT models
trained_models_folder = os.environ.get('HIT_PRETRAINED_FOLDER', os.path.join(repo_directory, 'pretrained'))  # folder to save the trained models
pretrained_male_smpl = os.path.join(trained_models_folder, 'pretrained_male_smpl.ckpt')
pretrained_female_smpl = os.path.join(trained_models_folder, 'pretrained_female_smpl.ckpt')
                                    
# Folder containing smpl/SMPL_MALE.pkl and smpl/SMPL_FEMALE.pkl, to download from https://smpl.is.tue.mpg.de/
smplx_models_path = os.environ.get('HIT_SMPL_MODELS', os.path.join(repo_directory, 'body_models'))

# Training logging
wandb_entity = 'mkeller' # wandb account or team to log to
wandb_project_name = 'hit_rel' # wandb project to log to
##################


# assets
v2p = os.path.join(package_directory, 'assets/v2p.pkl')  # file that for each smpl vertex, gives the corresponding part

n_chunks_test = 10 # number of chunks to split the test set into. Increase to reduce memory usage when caching the dataset
n_chunks_train = 20 # Same for the train set
    
# Logger for training
logger = 'wandb' # only wandb is supported
