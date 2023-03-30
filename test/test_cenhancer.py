import unittest

import torch

from mllib.src.audio import amplify_torch
from mllib.src.evaluate import stft_custom, istft_custom
from mllib.src.utils import load_yaml

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR

from mllib.src.ha.amplifier import NALRTorch
from mllib.src.ha.compressor import CompressorTorch

class ClarityEnhancerSanityCheck(unittest.TestCase):
    def test_nalr(self):
        """
        python -m unittest -v test.test_cenhancer.ClarityEnhancerSanityCheck.test_nalr
        """
        print()
        import json
        import librosa
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        from omegaconf import OmegaConf
        from scipy.io import wavfile
        from recipes.icassp_2023.MLbaseline.evaluate import make_scene_listener_list

        # Should put root path for dataset
        cfg = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")

        with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
            scenes_listeners = json.load(fp)

        with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        
        scenes_folder = Path(cfg.path.scenes_folder)
        
        scene_listener_pairs = make_scene_listener_list(
            scenes_listeners, cfg.evaluate.small_test
        )        
        
        scene, listener = scene_listener_pairs[0]
        
        audiogram = listener_audiograms[listener]

        # Read signals
        fs_signal, signal = wavfile.read(
            scenes_folder / f"{scene}_target_CH1.wav"
        )
        fs_ref_anechoic, ref_anechoic = wavfile.read(
            scenes_folder / f"{scene}_target_anechoic_CH1.wav"
        )
        fs_ref_target, ref_target = wavfile.read(
            scenes_folder / f"{scene}_target_CH1.wav"
        )

        signal = signal / 32768.0
        ref_anechoic = ref_anechoic / 32768.0
        ref_target = ref_target / 32768.0

        signal = signal[:4*fs_signal, ...]
        ref_anechoic = ref_anechoic[:4*fs_ref_anechoic, ...]
        ref_target = ref_target[:4*fs_ref_target, ...]

        assert fs_ref_anechoic == fs_ref_target

        rms_target = np.mean(ref_target**2, axis=0) ** 0.5
        rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
        ref = ref_anechoic * rms_target / rms_anechoic
        
        # hasqi_score = compute_metric(hasqi_v2_be, amplified, ref, audiogram, fs_signal)
        enhancer = NALR(**cfg.nalr)
        compressor = Compressor(**cfg.compressor) 

        enhancer_torch = NALRTorch(**cfg.nalr)
        compressor_torch = CompressorTorch(**cfg.compressor) 
        
        print(audiogram)
        
        cfs = np.array(audiogram["audiogram_cfs"])
        audiogram = np.array([audiogram[f"audiogram_levels_l"], 
                            audiogram[f"audiogram_levels_r"]])
        
        nalr_fir_left, _ = enhancer.build(audiogram[0], cfs)
        nalr_fir_right, _ = enhancer.build(audiogram[1], cfs)
        out_l = enhancer.apply(nalr_fir_left, signal[..., 0])
        out_r = enhancer.apply(nalr_fir_right, signal[..., 1])

        signal_torch = torch.from_numpy(signal)
        nalr_fir_left_torch = enhancer_torch.build(audiogram[0], cfs)
        nalr_fir_right_torch = enhancer_torch.build(audiogram[1], cfs)
        signal_torch = signal_torch.unsqueeze(0).unsqueeze(0)
        out_l_torch = enhancer_torch.apply(nalr_fir_left_torch, signal_torch[..., 0])
        out_r_torch = enhancer_torch.apply(nalr_fir_right_torch, signal_torch[..., 1])

        print(out_l.shape, out_l_torch.squeeze(0).squeeze(0).shape)
        print("Max error rate:", np.max(np.abs(out_l-out_l_torch.numpy())))

        out_l, _, _ = compressor.process(out_l)
        if cfg.soft_clip:
            out_l = np.tanh(out_l)

        out_l_torch = compressor_torch.process(out_l_torch)
        if cfg.soft_clip:
            out_l_torch = torch.tanh(out_l_torch)

        print(out_l.shape, out_l_torch.squeeze(0).squeeze(0).shape)
        print("Max error rate:", np.max(np.abs(out_l-out_l_torch.numpy())))            


        # [TODO] fft convolve
        # config = load_yaml("./test/conf/config.yaml")
        
        # # convert fft filters
        # nalr_fir = np.stack([nalr_fir_left, nalr_fir_right], axis=0)
        # # [TODO] Center nalr_fir or change nalr nfft
        # nalr_fir_fft = np.fft.rfft(nalr_fir, n=config.nalr.n_fft) # /cfg.nalr.nfir        
        # nalr_fir_fft = np.stack([nalr_fir_fft.real, nalr_fir_fft.imag], axis=-1)
        # nalr_fir_fft = torch.from_numpy(nalr_fir_fft)
        # nalr_fir_fft = nalr_fir_fft.unsqueeze(dim=-2).unsqueeze(dim=0)
        # # stft
        # signal_stft = stft_custom(torch.from_numpy(signal.T)[None],
        #                          config.nalr)
    
        # print(nalr_fir_fft.dtype, nalr_fir_fft.shape, signal_stft.shape, signal_stft.dtype)

        # buffer = signal_stft[..., 0]*nalr_fir_fft[..., 0] - signal_stft[..., 1]*nalr_fir_fft[..., 1]
        # signal_stft[..., 1] = signal_stft[..., 0]*nalr_fir_fft[..., 1] + signal_stft[..., 1]*nalr_fir_fft[..., 0]
        # signal_stft[..., 0] = buffer
        
        # print(signal_stft.shape)
        # # istft
        # isignal_stft = istft_custom(signal_stft, 
        #                             length=signal.shape[0], 
        #                             config=config.nalr)
        # isignal_stft = isignal_stft.squeeze(dim=0)
        # print(isignal_stft[0, ...].shape, signal[..., 0].shape, out_l.shape)
        
        # # [TODO] compare fir and fft result
        # out_l, _, _ = compressor.process(out_l)
        # if cfg.soft_clip:
        #     out_l = np.tanh(out_l)

        # # [TODO] Compress torch
        # out_signal_torch, = compressor.process_torch(isignal_stft)
        # if cfg.soft_clip:
        #     out_signal_torch = torch.tanh(out_signal_torch)

        # # [TODO] compare fir and torch result

    def test_amplify_torch(self):
        """
        python -m unittest -v test.test_cenhancer.ClarityEnhancerSanityCheck.test_amplify_torch
        """
        print()
        import json
        import librosa
        import numpy as np
        import soundfile as sf
        from pathlib import Path
        from omegaconf import OmegaConf
        from scipy.io import wavfile
        from recipes.icassp_2023.MLbaseline.evaluate import make_scene_listener_list

        # Should put root path for dataset
        cfg = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")

        with open(cfg.path.scenes_listeners_file, "r", encoding="utf-8") as fp:
            scenes_listeners = json.load(fp)

        with open(cfg.path.listeners_file, "r", encoding="utf-8") as fp:
            listener_audiograms = json.load(fp)
        
        scenes_folder = Path(cfg.path.scenes_folder)
        
        scene_listener_pairs = make_scene_listener_list(
            scenes_listeners, cfg.evaluate.small_test
        )        
        
        scene, listener = scene_listener_pairs[0]
        
        audiogram = listener_audiograms[listener]

        print(audiogram)

        # Read signals
        fs_signal, signal = wavfile.read(
            scenes_folder / f"{scene}_target_CH1.wav"
        )

        signal = signal / 32768.0

        signal = signal[:4*fs_signal, ...]
        signal = signal.astype(np.float32)
        batch = torch.from_numpy(signal.T)
        batch = torch.stack([batch, batch], dim=0) # batch, nchannel, nsamples
        batch = batch.unsqueeze(1) # batch, nspk, nchannel, nsamples
        enhancer = NALRTorch(**cfg.nalr)
        compressor = CompressorTorch(**cfg.compressor) 
        signal_amplfied = amplify_torch(signal=batch, 
                                        enhancer=enhancer,
                                        compressor=compressor,
                                        audiogram=audiogram,
                                        soft_clip=True)
