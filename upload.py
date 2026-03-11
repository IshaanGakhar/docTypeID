import kagglehub

handle = 'IshaanPerssonify/EPONAMix-Inference-Predicted-Relevant'
local_dataset_dir = '/home/ishaan/work/docFilter/EPONAMix-Inference/Predicted_Relevant'

# Create a new dataset
kagglehub.dataset_upload(handle, local_dataset_dir)