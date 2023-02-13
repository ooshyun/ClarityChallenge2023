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
    # main("./mllib/src/conf/config.yaml")

    # 2. Inference model
    # samples only including target's period, PIT
    model_path = "./result/conv-tasnet/20230207-184607"
    # # including all samples, PIT
    # model_path = "./result/conv-tasnet/20230207-185011"
    # # including all samples and no PIT
    # model_path = "./result/conv-tasnet/20230208-175200"

    name="_"+model_path.split("/")[-1]

    # # # 2.1 Get metric for trained model
    pretrained_model = model_path + "/config.yaml"
    main(path_config=pretrained_model, mode="test")

    # # 2.2 Get amplified signal and evaluate haspi/hasqi
    # config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")
    
    # # enhance
    # enhance(config, model_path=model_path, name=name)

    # # evaluate
    # run_calculate_si(config, name=name)

    # # report score
    # report_score(config, name=name)

    # # 2.3. Evaluate and submit Clarity challenge
    # config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config_eval.yaml")
    
    # # enhance
    # make_submission_clarity_challenge(config, model_path=model_path, name=name)