import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    def __init__(self, model, r=8):
        super(LoRA, self).__init__()
        self.model = model
        self.r = r
        self.freeze_model()
        self.lora_layers = nn.ParameterDict()  # Use ParameterDict instead of dict
        self.apply_lora()

    def forward(self, x):
        return self.model(x)
    
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def apply_lora(self, alpha=1.0):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv" in name:
                    print("Module in_features", module.in_features)
                    print("Module out_features", module.out_features)
                    # Create low-rank matrices      
                    lora_A = nn.Parameter(torch.zeros(self.r, module.in_features))
                    lora_B = nn.Parameter(torch.zeros(module.out_features, self.r))
                
                    # Initialize them
                    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                    nn.init.zeros_(lora_B)

                    # Replace all the . in name with _
                    name = name.replace(".", "_")
                    
                    # Store these parameters properly
                    self.lora_layers[f"{name}_A"] = lora_A
                    self.lora_layers[f"{name}_B"] = lora_B
                
                # Create a forward hook to modify the output
                def forward_hook(module, input, output, name=name):
                    if "qkv" in name:
                        lora_A = self.lora_layers[f"{name}_A"]
                        lora_B = self.lora_layers[f"{name}_B"]
                        lora_output = input[0] @ (lora_A.T @ lora_B.T)
                        return output + alpha * lora_output
                    else:
                        return output
                
                module.register_forward_hook(forward_hook)
