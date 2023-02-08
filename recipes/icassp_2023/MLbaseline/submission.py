""" Run the dummy enhancement. """
import json
import logging
import pathlib

import copy
# import hydra
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
def make_submission_clarity_challenge(cfg: DictConfig, model_path) -> None:
    submission_folder = pathlib.Path("./ICASSP_E009")
    submission_folder.mkdir(parents=True, exist_ok=True)

    scenes_listeners_file_list = OmegaConf.to_object(cfg.path.scenes_listeners_file)
    scenes_folder_list =  OmegaConf.to_object(cfg.path.scenes_folder)
    for icase, case in enumerate(cfg.path.scenes_cases):
        """Run the dummy enhancement."""
        assert case in cfg.path.scenes_listeners_file[icase] and case in cfg.path.scenes_folder[icase], "scene folder or scene listers file name is wrong..."
        
        enhanced_folder = pathlib.Path(submission_folder / case)
        enhanced_folder.mkdir(parents=True, exist_ok=True)

        print(f"Process {enhanced_folder.as_posix()} ...")

        scenes_listeners_file = pathlib.Path(cfg.path.metadata_dir) / scenes_listeners_file_list[icase]
        scenes_folder = pathlib.Path(cfg.path.root) / scenes_folder_list[icase]

        with open(scenes_listeners_file, "r", encoding="utf-8") as fp:
            scenes_listeners = json.load(fp)

        # Make list of all scene listener pairs that will be run
        scene_listener_pairs = make_scene_listener_list(
            scenes_listeners, cfg.evaluate.small_test
        )

        # Load ML model
        config = load_yaml(model_path + "/config.yaml")
        model = get_model(config.model)
        config.solver.resume = model_path
        # config.solver.preloaded_model = model_path + " " # specific model path
        solver = Solver(config=config, model=model)
        pretrained_model = copy.deepcopy(solver.model)
        device = solver.device
        del solver
        
        count_file = 0
        for scene, listener in tqdm(scene_listener_pairs):
            sample_freq, signal = wavfile.read(
                scenes_folder / f"{scene}_mix_CH1.wav"
            )

            # Convert to 32-bit floating point scaled between -1 and 1
            signal = (signal / 32768.0).astype(np.float32)

            # Enhance using ML
            signal = np.transpose(signal, axes=[1, 0])
            length = signal.shape[-1]
            signal = torch.from_numpy(signal)
            
            
            if config.dset.sample_rate != sample_freq:
                signal = julius.resample.resample_frac(signal, sample_freq, config.dset.sample_rate)

            nchannel, nsample = signal.shape
            
            if not config.optim.pit:
                signal = torch.reshape(signal, shape=(nchannel, 1, nsample))
            
            enhanced = evaluate(mixture=signal[None], 
                            model=pretrained_model, 
                            device=device, 
                            config=config)
            if config.optim.pit:
                index_enhanced = 0 # evaluated by training dataset
                signal = enhanced[:, index_enhanced, ...]
            else:
                signal = enhanced

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

        print(f"\t {count_file} files done in {enhanced_folder.as_posix()}...")