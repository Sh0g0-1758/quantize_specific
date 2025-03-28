__version__ = "0.0.1"
__author__ = 'vision and language group'
__credits__ = 'IITR'

from dataclasses import dataclass
from typing import Optional, Dict, List
import argparse
from typing_extensions import Literal
import torch
import numpy as np
import random

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)

logging.set_verbosity_error()


@dataclass
class QuantConfig:
    load_in_4bit: Optional[bool]
    load_in_8bit: Optional[bool]
    bnb_4bit_use_double_quant: Optional[bool]
    bnb_4bit_quant_storage: Optional[torch.dtype]
    bnb_4bit_compute_dtype: Optional[torch.dtype]
    bnb_4bit_quant_type: Optional[str] = None


def create_dtype_map() -> Dict[str, torch.device]:
    mapping = {
        ("float16", "fp16"): torch.float16,
        ("bfloat16",): torch.bfloat16,
        ("float32", "fp32"): torch.float32,
    }
    dtype_map = {}
    for keys, value in mapping.items():
        for key in keys:
            dtype_map[key] = value
    return dtype_map


def generate_decoder_map():
    decoder_map: Dict[str, str] = {
        "llama": "model.layers",
        "gpt_neox": "gpt_neox.layers",
        "mistral": "model.layers",
        "mixtral": "model.layers",
    }
    return decoder_map


DecoderMap = generate_decoder_map()


def set_seed(seed: int = 42):
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # PyTorch GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    # Avoid non-deterministic optimizations
    torch.backends.cudnn.benchmark = False

##
#  The library essntially creates two models.
#  It uses quant_config as the configuration for the first model and quant_config_swap as the configuration for the second model.
#  The library then allows the user to give either a layer_swap_config or a swap_every configuration.
#  The layer_swap_config is a list of layers that the user wants to swap between the two models.
#  The swap_every configuration is a list of fractions that the user wants to swap between the two models. It takes a string of the form
#  'x/y' where x and y are positive integers and represents the portion of layers that the user wants to swap ie. we divide the total
#  number of layers into y parts and swap the xth part.
##
class MemorizationAnalyser:
    def __init__(
        self,
        model_name: str,
        model_family: str,
        quant_config: QuantConfig,
        quant_config_swap: Optional[QuantConfig],
        layer_swap_config: Optional[List[int]],
        swap_every: Optional[List[str]],
        device_map: Literal["cpu", "auto", "balanced"] = "balanced",
        dtype_map: Dict = create_dtype_map(),
    ):
        if layer_swap_config is not None and swap_every is not None:
            raise ValueError(
                f"Please specify only one of layer_swap_config or swap_every")

        self.device_map = device_map

        self.dtype = dtype_map[quant_config.bnb_4bit_compute_dtype]

        self.load_in = "fp32"
        if quant_config.load_in_4bit:
            self.load_in = "4bit/" + quant_config.bnb_4bit_quant_type
            if quant_config.bnb_4bit_use_double_quant:
                self.load_in += "/double"
            else:
                self.load_in += "/single"

        elif quant_config.load_in_8bit:
            self.load_in = "8bit"

        if quant_config.bnb_4bit_quant_type:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.load_in_4bit,
                bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
                load_in_8bit=quant_config.load_in_8bit,
                bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self.dtype
            )
        else:
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.load_in_4bit,
                load_in_8bit=quant_config.load_in_8bit,
                bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=self.dtype
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quant_config,
            device_map=self.device_map
        )
        self.model.eval()

        if quant_config_swap and (layer_swap_config or swap_every):
            self.dtype_swap = dtype_map[quant_config_swap.bnb_4bit_compute_dtype]

            if quant_config.bnb_4bit_quant_type:
                self.quant_config_swap = BitsAndBytesConfig(
                    load_in_4bit=quant_config.load_in_4bit,
                    bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
                    load_in_8bit=quant_config.load_in_8bit,
                    bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=self.dtype
                )
            else:
                self.quant_config_swap = BitsAndBytesConfig(
                    load_in_4bit=quant_config.load_in_4bit,
                    load_in_8bit=quant_config.load_in_8bit,
                    bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=self.dtype
                )

            if quant_config_swap.load_in_4bit:
                self.load_in_swap = "4bit/" + quant_config_swap.bnb_4bit_quant_type
                if quant_config_swap.bnb_4bit_use_double_quant:
                    self.load_in_swap += "/double"
                else:
                    self.load_in_swap += "/single"

            elif quant_config_swap.load_in_8bit:
                self.load_in_swap = "8bit"

            self.model_swap = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.quant_config_swap,
                device_map=self.device_map
            )
            self.model_swap.eval()

            try:
                decoders_1 = eval(f"self.model.{DecoderMap[model_family]}")
                decoders_2 = eval(
                    f"self.model_swap.{DecoderMap[model_family]}")
            except AttributeError as e:
                raise ValueError(f"Unsupported model family '{model_family}' \
                        or invalid layer attribute: {e}")

            if layer_swap_config:
                self.layer_swap_config = layer_swap_config
                for layer in self.layer_swap_config.skip_layers:
                    decoders_1[layer] = decoders_2[layer]

            elif swap_every:
                for swap in swap_every:
                    try:
                        swap = swap.strip()
                        for swap in swap.split():
                            num, denom = map(int, swap.split("/"))
                            for layer in range(len(decoders_1)):
                                if (layer + 1) % denom == num % denom:
                                    decoders_1[layer] = decoders_2[layer]
                    except ValueError as e:
                        raise ValueError("swap_every must be in the format 'x/y' \
                                        where x and y are positive integers: {e}")

        if (quant_config_swap is None and (layer_swap_config is not None or swap_every is not None)) or \
                (quant_config_swap is not None and layer_swap_config is None and swap_every is None):
            raise ValueError(f"Please provide both quant_config_swap and either layer_swap_config or swap_every,\
                            but not both.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze degree of memorization for a specified model and dataset, \
            varying quantization parameters"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to the model JSON configuration file"
    )
    parser.add_argument(
        "--quant-config",
        type=str,
        required=True,
        help="Path to the quantization JSON configuration file"
    )
    parser.add_argument(
        "--quant-config-swap",
        type=str,
        help="Path to the swap quantization JSON configuration file"
    )
    parser.add_argument(
        "--layer-swap-config",
        type=str,
        help="Path to the layer swap JSON configuration file"
    )
    parser.add_argument(
        "--swap-every",
        type=str,
        nargs="+",
        default=["3/4", "4/4"],
        help="Specify which fraction of decoder layers to quantize e.g ['3/4', '4/4'] \
                will quantize the third and fourth quarter of decoders"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="BitsAndBytes Device Map configuration"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="common fixed seed for all libraries"
    )

    args = parser.parse_args()
    set_seed(args.seed)
