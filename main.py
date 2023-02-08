from mllib.src.train import (
    main
)

from omegaconf import OmegaConf
from recipes.icassp_2023.MLbaseline.enhance import enhance
from recipes.icassp_2023.MLbaseline.evaluate import run_calculate_si
from recipes.icassp_2023.MLbaseline.report_score import report_score

from recipes.icassp_2023.MLbaseline.submission import make_submission_clarity_challenge

if __name__=="__main__":
    # 1. Train
    main("./mllib/src/conf/config.yaml")

    # model_path = "/home/daniel0413/workplace/project/SpeechEnhancement/SpeechEnhancementHL-Clarity/result/conv-tasnet/20230207-080249"

    # # 2. Develop Clarity challenge
    # config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")
    
    # # enhance
    # enhance(config, model_path=model_path)

    # # evaluate in development
    # # evaluate
    # run_calculate_si(config)

    # # report score
    # report_score(config)

    # # 3. Evaluate and submit Clarity challenge
    # config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config_eval.yaml")
    
    # # enhance
    # make_submission_clarity_challenge(config, model_path=model_path)