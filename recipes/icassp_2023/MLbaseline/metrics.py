import json
import pathlib
import soundfile as sf
import random
import numpy as np
import warnings

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.evaluator.haspi import haspi_v2_be
from clarity.evaluator.hasqi import hasqi_v2_be

from .evaluate import (
    amplify_signal, 
    compute_metric,
)


def get_amplified_signal(signal, sample_rate, cfg):
    assert signal.shape[-1] == 2, "Signal should be stereo..."
    assert sample_rate == cfg.nalr.fs
    assert isinstance(signal, np.ndarray)

    enhancer = NALR(**cfg.nalr)
    compressor = Compressor(**cfg.compressor)
    
    # amplify left and right ear signals
    with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
        listener_audiograms = json.load(fp)
        
    nlistener = 1
    listener_list = list(listener_audiograms.keys())
    scenes_listeners = random.sample(listener_list, nlistener)

    amplified_scene_list = []
    for listener in scenes_listeners:

        audiogram = listener_audiograms[listener]

        out_l = amplify_signal(signal[:, 0], audiogram, "l", enhancer, compressor)
        out_r = amplify_signal(signal[:, 1], audiogram, "r", enhancer, compressor)
        amplified = np.stack([out_l, out_r], axis=1)

        if cfg.soft_clip:
            amplified = np.tanh(amplified)
        
        amplified_scene_list.append({'audiogram': audiogram, 'amplified': amplified})
        
    return amplified_scene_list

def get_ref(scene, cfg):

    scenes_folder = pathlib.Path(cfg.path.scenes_folder)

    ref_anechoic, fs_ref_anechoic = sf.read(
        scenes_folder / f"{scene}_target_anechoic_CH1.wav"
    )
    ref_target, fs_ref_target = sf.read(
        scenes_folder / f"{scene}_target_CH1.wav"
    )

    assert fs_ref_anechoic == fs_ref_target

    rms_target = np.mean(ref_target**2, axis=0) ** 0.5
    rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
    ref = ref_anechoic * rms_target / rms_anechoic    

    return ref, fs_ref_target

def compute_hasqi(amplified, reference, audiogram, sample_rate):
    hasqi_score = compute_metric(hasqi_v2_be, amplified, reference, audiogram, sample_rate)
    return hasqi_score
        
def compute_haspi(amplified, reference, audiogram, sample_rate):
    haspi_score = compute_metric(haspi_v2_be, amplified, reference, audiogram, sample_rate)
    return haspi_score


def evaluate_clarity(scene, enhanced, sample_rate, cfg):
    ch, nsample = enhanced.shape
    enhanced = np.reshape(enhanced, newshape=(nsample, ch))

    reference, fs_ref = get_ref(scene=scene, cfg=cfg)
    
    if enhanced.shape[0] != reference.shape[0]:
        # Usually the difference is 1 sample
        # warnings.warn(f"Scene {scene}: {enhanced.shape}, {reference.shape}, shape is different...")
        if enhanced.shape[0] > reference.shape[0]:
            enhanced = enhanced[:reference.shape[0], ...]
        elif enhanced.shape[0] < reference.shape[0]:
            reference = reference[:enhanced.shape[0], ...]

    assert fs_ref == sample_rate, f'{sample_rate}, {fs_ref}, Sample rate is different...'

    # currently only choose only 1 listeners, dev give 3 listeners
    amplified_scene_list = get_amplified_signal(enhanced, sample_rate=sample_rate, cfg=cfg)

    score = np.zeros(shape=(len(amplified_scene_list), 2), dtype=np.float32)
    for icase, amplified_scene in enumerate(amplified_scene_list):
        hasqi = compute_haspi(amplified_scene['amplified'], reference, amplified_scene['audiogram'], sample_rate=sample_rate)
        haspi = compute_hasqi(amplified_scene['amplified'], reference, amplified_scene['audiogram'], sample_rate=sample_rate)
        score[icase] = hasqi, haspi
    return score

