import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(device)
    print('Running on GPU: {}'.format(gpu_name))
else:
    print('Running on CPU')
