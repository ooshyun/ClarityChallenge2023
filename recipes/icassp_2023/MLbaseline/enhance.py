""" Run the dummy enhancement. """
import json
import logging
import pathlib

import copy
import hydra
import numpy as np
from .evaluate import make_scene_listener_list
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm import tqdm

logger = logging.getLogger(__name__)

import torch
import julius
from mllib.src.evaluate import evaluate
from mllib.src.utils import load_yaml
from mllib.src.distrib import get_model
from mllib.src.solver import Solver

# @hydra.main(config_path=".", config_name="config")
def enhance(cfg: DictConfig) -> None:
    """Run the dummy enhancement."""
    enhanced_folder = pathlib.Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
        scenes_listeners = json.load(fp)

    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)  # noqa: F841

    # Make list of all scene listener pairs that will be run
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, cfg.evaluate.small_test
    )

    # Load ML model
    model_path = "/home/daniel0413/workplace/project/SpeechEnhancement/SpeechEnhancementHL/result/dcunet/20230128-234556"
    config = load_yaml(model_path + "/config.yaml")
    model = get_model(config.model)
    config.solver.resume = model_path
    # config.solver.preloaded_model = model_path + " " # specific model path
    solver = Solver(config=config, model=model)
    pretrained_model = copy.deepcopy(solver.model)
    device = solver.device
    del solver
    
    for scene, listener in tqdm(scene_listener_pairs):
        sample_freq, signal = wavfile.read(
            pathlib.Path(cfg.path.scenes_folder) / f"{scene}_mix_CH1.wav"
        )

        # Convert to 32-bit floating point scaled between -1 and 1
        signal = (signal / 32768.0).astype(np.float32)

        # # Audiograms can read like this, but they are not needed for the baseline
        #
        # cfs = np.array(listener_audiograms[listener]["audiogram_cfs"])
        #
        # audiogram_left = np.array(
        #    listener_audiograms[listener]["audiogram_levels_l"]
        # )
        # audiogram_right = np.array(
        #    listener_audiograms[listener]["audiogram_levels_r"]
        # )

        # Baseline just reads the signal from the front microphone pair
        # and write it out as the enhanced signal
        
        # Enhance using ML
        signal = np.transpose(signal, axes=[1, 0])
        length = signal.shape[-1]
        signal = torch.from_numpy(signal)
        
        if config.dset.sample_rate != sample_freq:
            signal = julius.resample.resample_frac(signal, sample_freq, config.dset.sample_rate)
        
        nchannel, nsample = signal.shape
        signal = torch.reshape(signal, shape=(nchannel, 1, nsample))
        
        signal = evaluate(mixture=signal, 
                        model=pretrained_model, 
                        device=device, 
                        config=config)

        signal = torch.reshape(signal, shape=(nchannel, nsample))
        
        if config.dset.sample_rate != sample_freq:
            signal = julius.resample.resample_frac(signal, config.dset.sample_rate, sample_freq, full=True)
        
        signal = signal.cpu().detach().numpy()
        if signal.shape[-1] != length:
            if signal.shape[-1] > length:
                signal = signal[..., :length]
            else:
                pad_signal = np.zeros(shape=(2, len(signal.shape)), dtype=int)
                pad_signal[-1] = length - signal.shape[-1]
                signal = np.pad(signal, pad_signal, mode='constant', constant_values=0)

        signal = np.transpose(signal, axes=[1, 0])
        wavfile.write(
            enhanced_folder / f"{scene}_{listener}_enhanced.wav", sample_freq, signal
        )

if __name__ == "__main__":
    enhance()
