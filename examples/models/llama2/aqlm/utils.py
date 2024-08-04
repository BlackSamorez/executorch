import torch
from torch import nn

from accelerate import init_empty_weights
from executorch.examples.models.llama2.aqlm.linear import Aqlm2x8Linear
    

def replace_with_aqlm_linear(
    model,
    quantization_config=None,
    linear_weights_not_to_quantize=None,
    current_key_name=None,
    has_been_replaced=False,
):
    """
    Public method that recursively replaces the Linear layers of the given model with AQLM quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successfull or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AqlmConfig`):
            The quantization config object that contains the quantization parameters.
        linear_weights_not_to_quantize (`list[str]`, *optional*):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """
    if linear_weights_not_to_quantize is None:
        linear_weights_not_to_quantize = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `linear_weights_not_to_quantize`
            if ".".join(current_key_name) + ".weight" not in linear_weights_not_to_quantize:
                # with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = Aqlm2x8Linear(
                        in_features,
                        out_features,
                        bias=module.bias is not None,
                        in_group_size=quantization_config.in_group_size,
                        out_group_size=quantization_config.out_group_size,
                        num_codebooks=quantization_config.num_codebooks,
                        nbits_per_codebook=quantization_config.nbits_per_codebook,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_aqlm_linear(
                module,
                quantization_config=quantization_config,
                linear_weights_not_to_quantize=linear_weights_not_to_quantize,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def transpose_codes(
    model,
    current_key_name=None,
    has_been_transposed=False,
):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, Aqlm2x8Linear):
            model._modules[name].transpose_codes()
            has_been_transposed = True
        if len(list(module.children())) > 0:
            _, has_been_transposed = transpose_codes(
                module,
                current_key_name=current_key_name,
                has_been_transposed=has_been_transposed,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_transposed
