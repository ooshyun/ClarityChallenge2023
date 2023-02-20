import argparse
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
      '--mode',
      help='Mode for procedure of ML model',
      required=False,
      default='train',
      type=str)

    parser.add_argument(
        '--config',
      help='Configuration for ML model',
      required=False,
      default="./mllib/src/conf/config.yaml",
      type=str)  

    parser.add_argument(
        '--model_path',
      help='Configuration for ML model',
      required=False,
      default="",
      type=str)  

    parser.add_argument(
        '--dev',
      help='Use development dataset',
      required=False,
      default=False,
      type=bool)  
    
    parser.add_argument(
        '--device',
      help='Device for ML training',
      required=False,
      default="cpu",
      type=str)  

    args = parser.parse_args()
    
    
    # model_path = "./result/conv-tasnet/20230207-184607" # samples only including target's period, PIT
    # model_path = "./result/conv-tasnet/20230207-185011" # including all samples, PIT
    # model_path = "./result/conv-tasnet/20230208-175200"  # including all samples and no PIT
    # model_path = "./result/conv-tasnet/20230216-113419"   # including all samples and no PIT for paper analyze
    # model_path = './result/conv-tasnet/20230219-184837'

    model_path = args.model_path

    if args.mode == "train":
        from mllib.src.train import main
        main(args.config)

    elif args.mode == "inference":
        assert len(model_path) > 0, f"{args.mode} Mode should have model path..."

        from mllib.src.train import main
        pretrained_model = model_path + "/config.yaml"
        main(obj_config=pretrained_model, mode="test", dev=args.dev, device=args.device, save=True)

    elif args.mode == "clarity_inference":
        assert len(model_path) > 0, f"{args.mode} Mode should have model path..."        
        
        from omegaconf import OmegaConf
        from recipes.icassp_2023.MLbaseline.enhance import enhance
        from recipes.icassp_2023.MLbaseline.evaluate import run_calculate_si
        from recipes.icassp_2023.MLbaseline.report_score import report_score
        name="_"+model_path.split("/")[-1]
        config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config.yaml")
        enhance(config, model_path=model_path, name=name)   # enhance
        run_calculate_si(config, name=name)                 # evaluate
        report_score(config, name=name)                     # report score

    elif args.mode == "clarity_report":
        assert len(model_path) > 0, f"{args.mode} Mode should have model path..."
        
        from omegaconf import OmegaConf
        from recipes.icassp_2023.MLbaseline.submission import make_submission_clarity_challenge
        name="_"+model_path.split("/")[-1]
        config = OmegaConf.load("./recipes/icassp_2023/MLbaseline/config_eval.yaml")
        make_submission_clarity_challenge(config, model_path=model_path, name=name) # enhance

    else:
        raise ValueError(f"Mode is validable (train, inference, clarity_inference, clarity_report)")
