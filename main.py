from mllib.src.train import (
    main
)

from omegaconf import OmegaConf
from recipes.icassp_2023.MLbaseline.enhance import enhance
from recipes.icassp_2023.MLbaseline.evaluate import run_calculate_si
from recipes.icassp_2023.MLbaseline.report_score import report_score

if __name__=="__main__":
    # Train
    # main("./mllib/src/conf/config.yaml")


    # Clarity challenge
    config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")
    
    # enhance
    enhance(config)

    # evaluate
    # run_calculate_si(config)

    # report score
    # report_score(config)
