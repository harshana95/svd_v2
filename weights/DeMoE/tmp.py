import torch

d = torch.load('./model.pt')
k1 ='models_list'
k2 ='modules_list'
kk = [k for k in d['params'].keys() if k1 in k]

for k in kk:
    d['params'][k.replace(k1, k2)] = d['params'].pop(k)
    print(f"{k.replace(k1, k2)} <- {k}")

print([k for k in d['params'].keys() if k1 in k])
torch.save(d, './model.pt')


# from safetensors.torch import load_file

# # 1. Load the dictionary from the file
# # By default, it loads to CPU. You can set device="cuda" if needed.
# sd = load_file("model.safetensors")
# breakpoint()

from safetensors.torch import save_file

# 1. Load the checkpoint
# map_location="cpu" ensures you don't need a GPU for the conversion
checkpoint = torch.load('./model.pt', map_location="cpu")

# 2. Extract the weights from the 'params' key
if "params" in checkpoint:
    state_dict = checkpoint["params"]
else:
    # Fallback in case the key is named differently or it's a direct state_dict
    state_dict = checkpoint

# 3. Save to safetensors format
# Note: state_dict must be a Dict[str, torch.Tensor]
save_file(state_dict, "./diffusion_pytorch_model.safetensors")
