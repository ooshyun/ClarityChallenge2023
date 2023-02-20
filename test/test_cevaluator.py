import unittest
from clarity.evaluator.haspi import haspi_v2_be
from clarity.evaluator.hasqi import hasqi_v2_be

class ClarityEvaluatorSanityCheck(unittest.TestCase):
    def test_ear_model(self):
        """
        python -m unittest -v test.test_cevaluator.ClarityEvaluatorSanityCheck.test_ear_model
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
        
        print(listener_audiograms)
        
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

        assert fs_ref_anechoic == fs_ref_target

        rms_target = np.mean(ref_target**2, axis=0) ** 0.5
        rms_anechoic = np.mean(ref_anechoic**2, axis=0) ** 0.5
        ref = ref_anechoic * rms_target / rms_anechoic
        
        # hasqi_score = compute_metric(hasqi_v2_be, amplified, ref, audiogram, fs_signal)
        signal = ref_anechoic
        score_haspi = haspi_v2_be(
            xl=ref[:, 0],
            xr=ref[:, 1],
            yl=signal[:, 0],
            yr=signal[:, 1],
            fs_signal=fs_signal,
            audiogram_l=audiogram["audiogram_levels_l"],
            audiogram_r=audiogram["audiogram_levels_r"],
            audiogram_cfs=audiogram["audiogram_cfs"],
        )

        score_hasqi = hasqi_v2_be(
            xl=ref[:, 0],
            xr=ref[:, 1],
            yl=signal[:, 0],
            yr=signal[:, 1],
            fs_signal=fs_signal,
            audiogram_l=audiogram["audiogram_levels_l"],
            audiogram_r=audiogram["audiogram_levels_r"],
            audiogram_cfs=audiogram["audiogram_cfs"],
        )

        print(score_haspi + score_hasqi, score_haspi, score_hasqi)
