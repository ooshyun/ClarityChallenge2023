# Notes for Speech enhancement with hearing aid

## 1. Inference
### 1.1 Inference SISDR/PESQ/ESTOI/HASPI/HASQI in Clarity Challenge
1. result/conv-tasnet/20230219-184837
    - test/result/consv-tasnet/20230321-154809

2. result/conv-tasnet/20230219-205507
    - test/result/consv-tasnet/20230321-171419

3. result/conv-tasnet/20230220-003039
    - test/result/conv-tasnet/20230330-103906

4. result/conv-tasnet/20230220-100114
    - test/result/consv-tasnet/20230321-224529

5. result/conv-tasnet/20230221-145148
    - test/result/conv-tasnet/20230330-102910

6. result/conv-tasnet/20230221-231507
    - test/result/conv-tasnet/20230322-105808

7. result/conv-tasnet/20230223-140053
    - test/result/conv-tasnet/20230322-114218

- 1.2 Inference SISDR/PESQ/ESTOI/HASPI/HASQI in VoiceBankDEMAND
    - result/conv-tasnet/20230219-184837
        - test/result/conv-tasnet/20230330-110519
        - X remove noise
        - Even index is difference between clean and noisy

## 2. Experiment
- ./result/conv-tasnet/20230207-184607
    ```
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,      3
    - cutting by target speech
    - PIT
    - haspi 0.243942, hasqi 0.145829
    ```

- [V] result/conv-tasnet/20230215-115103
    ```
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,      3
    - no cutting by target speech
    - PIT
    - skip: False
    - terriable
    - haspi 0.196068, hasqi 0.116452(skip: False, batch 4)
    ```

- [V] result/conv-tasnet/20230215-222157
    ```
    - 1 epoch, 4800, 1:00
    - model path
        1. result/conv-tasnet/20230215-115521
        2. result/conv-tasnet/20230215-222157
        3. result/conv-tasnet/20230216-113419
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,      3
    - no cutting by target speech
    - no PIT
    - skip: False
    - haspi 0.240524, hasqi 0.146839(skip: False)
    - No PIT has speaker tracining issue?
    - [TODO] Why PESQ is 0?
    - [V] python main.py --mode inference --model_path ./result/conv-tasnet/20230216-113419 --dev True --device gpu
        - result/conv-tasnet/20230218-221903
    - [V] python main.py --mode inference --model_path ./result/conv-tasnet/20230216-113419 --device cpu
        - result/conv-tasnet/20230218-213304
    ```

- [V] result/conv-tasnet/20230216-211924
    ```
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Causal, batch
    - 256 	16 	128 256  3  8  3   gLN    X 	 3 -> 2
    - no cutting by target speech
    - no PIT
    - clip_grad_norm 5
    - result/conv-tasnet/20230216-211924 out of memory CUDA
    - result/conv-tasnet/20230216-221404 batch 3 -> 2 - 지워버림 ㅠ
    - train data 2000 -> sound X
    ```
- [V] result/conv-tasnet/20230217-141959
    ```
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 16, 128, 256, 3, 7, 3,  gLN,   X,      3
    - no cutting by target speech
    - PIT
    - skip: False
    - result/conv-tasnet/20230217-110414
    - result/conv-tasnet/20230217-141959 -> computer power off
    - result/conv-tasnet/20230218-111054
    - [?] R 3 -> 2? How to increase batch?
    ```

- [STOP] result/conv-tasnet/20230218-221830
    ```
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,      16
    - no cutting by target speech
    - no PIT
    - skip: False
    - segment 4
    ```

-  [STOP] result/conv-tasnet/20230219-012355
    ```
    - GPU 17.3G (batch 16: 18.3G)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 16, 128, 256, 3, 7, 3,  gLN,   X,      4
    - no cutting by target speech
    - no PIT
    - skip: False
    - segment 4(1 -> 4 -> the converstion of loss step is faster(Maybe L?))
    - ./result/conv-tasnet/20230219-012355 -> cuda out of memory
    - result/conv-tasnet/20230219-095027 -> 12 epoch 이후 수렴변화가 적음, sound도 noise 제거가 안됌
    ```

-  [STOP] result/conv-tasnet/20230219-115449
    ```
    - 1 epoch(train/validation): 30 min
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 16, 128, 256, 3, 7, 3,  gLN,   X,      4
    - [V] segment 1 -> 4 -> loss step에서도 수렴 speed도 빨라졌어(이건 L 때문)
        - [TODO] souce code 측면에서 reshpae, stack, pad, view, concat는 어떻게 할까?
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    ```

- [STOP] result/conv-tasnet/20230219-151050
    ```
    - 1 epoch: 316+56 step
    - 1 epoch : 2 min -> 200 epoch * 2 = 400 min = 6.5H
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - PIT -> spk0, spk1이랑 같음 -> PIT 방식이 잘못됌(summation의 최소인데, 조합 중 한가지의 loss가 최소를 고름)
    - skip: False
    - segment 4
    - dataset channel 1 (except 0, 2, 3)
    ```
- [GOOD] result/conv-tasnet/20230219-184837
    ```
    - 1 epoch: 316+56 step
    - 1 epoch : 2 min -> 200 epoch * 2 = 400 min = 6.5H
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - no PIT
    - skip: False
    - segment 4
    - dataset channel 1 (except 0, 2, 3)
    - [TODO] Test channel 0, 2, 3
    - [TODO] Test testset and devset -> find out why PIT needs

    - [V] python main.py --mode inference --model_path ./result/conv-tasnet/20230219-184837 --dev True --device cpu
        - ./test/result/conv-tasnet/20230219-212552
    ```

- [GOOD] result/conv-tasnet/20230219-205507
    ```
    - 1 epoch: 316+56 step
    - 1 epoch : 2 min -> 200 epoch * 2 = 400 min = 6.5H
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             -> expectation SISDR: 13
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 1 (except 0, 2, 3)
    - PIT loss mean X
    - python main.py --mode inference --model_path ./result/conv-tasnet/20230219-205507 --dev True --device cpu
        - ./test/result/conv-tasnet/20230220-194808
    ```

- [GOOD] result/conv-tasnet/20230220-003039
    ```
    - 1 epoch: 316+56 step
    - 1 epoch : 2 min -> 200 epoch * 2 = 400 min = 6.5H
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 1 and (except 0, 2, 3)
    - target anarchoic
    - PIT loss mean 
    - python main.py --mode inference --model_path ./result/conv-tasnet/20230220-003039 --dev True --device cpu
        - ./test/result/conv-tasnet/20230220-105225
    ```

- [GOOD] result/conv-tasnet/20230220-100114
    ```
    - 1 epoch: 316+56 step
    - 1 epoch : 2 min -> 200 epoch * 2 = 400 min = 6.5H
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 0, 1, 2, 3
    - PIT loss mean O
    ```

- [X] result/conv-tasnet/20230220-231408 loss did not change
    ```
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 1 (except 0, 2, 3)
    - se model: ./result/conv-tasnet/20230219-205507
    - [NEW] deverberation with amplfied model
        - wav -> denoising model -> enhanced wav -> deverb -> denorm(z-score) -> amplify -> norm(z-score) -> loss
        - dataset: target is anarchic wav 
        - result: loss is tooooo small: grad 0
    - [NEXT] Fine tune se model with anarchic wav -> loss did not change(20230220-231408)
    ```

- [SOSO] ./result/conv-tasnet/20230221-145148
    ```
    - ./result/conv-tasnet/20230221-101216
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 128, 40, 128, 256, 3, 7, 2,  gLN,   X,     16             
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 0, 1, 2, 3
    - norm X
    ```

- [GOOD] ./result/conv-tasnet/20230221-231507
    ```
    - GPU 18.8G(L=20)
    - GPU 17.9G(L=40)
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 512, 	32 	128, 512  3  8  3   gLN    X 	  4
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 0, 1, 2, 3
    - norm z-score
    - sounds different from N=128
    - max가 3배정도 커져서 나왔다
    ```

- [GOOD] ./result/conv-tasnet/20230223-140053
    ```
    - GPU 18.8G(L=20)
    - GPU 17.9G(L=40)
    - Reference. https://github.com/JusperLee/Conv-TasNet/tree/9eac70d28a5dba61172ad39dd9fb90caa0d1a45f
    - difference: decoder(transposed convolution and overlap)
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 512, 	40 	128, 512  3  8  4   gLN    X 	  4
    - randomly cropping the wavform
    - PIT
    - skip: False
    - segment 4
    - dataset channel 0, 1, 2, 3
    - norm z-score
    ```

- [SOSO] ./result/rnn-stft-mask/20230228-122947
    ```
    - rnn_type: 'lstm'
    - rnn_hidden: 896
    - rnn_layer: 3
    - activation: "relu"
    - drop_out: 0.5
    - bidirectional: True
    - MSE    
    - PIT
    - segment 4
    - dataset channel 0, 1, 2, 3
    - norm z-score
    ```

- [X] ./result/rnn-stft-mask/20230306-101834
    ```
    - rnn_type: 'lstm'
    - rnn_hidden: 1792
    - rnn_layer: 2
    - activation: "relu"
    - drop_out: 0.5
    - bidirectional: False
    - MSE
    - PIT
    - segment 4
    - dataset channel 0, 1, 2, 3
    - norm z-score
    ```

- [X] GPU 18.8G
    - 	N, 	L, 	 B,   H, P, X, R, Norm, Casual, batch
    - 256 	20 	256 512  3  8  4   gLN    X 	  3 -> validation cuda memory

## 3. Notes
### Research
- channel 1, 2, 3
    - A hearing aid with 3 microphone inputs (front, mid, rear). The hearing aid has a Behind-The-Ear (BTE) form factor; see Figure 1. The distance between microphones is approx. 7.6 mm. The properties of the tube and ear mould are not considered.
    - ch 0: Close to the eardrum.
    - anechoic target reference (front microphone).

- total train validation test scene number
    - total: 84000 = 6000 set
    - train validation test [0.85, 0.1, 0.05]
    - train validation test [5100, 600, 300]

- total dev scene number
    - total: 35000 = 2500 set
           
    - 1 set = 14 
       - S06001_hr.wav
       - S06001_interferer_CH0.wav
       - S06001_interferer_CH1.wav
       - S06001_interferer_CH2.wav
       - S06001_interferer_CH3.wav
       - S06001_mix_CH0.wav
       - S06001_mix_CH1.wav
       - S06001_mix_CH2.wav
       - S06001_mix_CH3.wav
       - S06001_target_CH0.wav
       - S06001_target_CH1.wav
       - S06001_target_CH2.wav
       - S06001_target_CH3.wav
       - S06001_target_anechoic_CH1.wav

- Paper research
    - tu: DHASPI
    - tammen: spk embedding + multichannel(CH1-Ch3) and conformer
    - ouyang: denoise + beamformer
    - liu: DRC Net, multi channel
    - lei: dereverb - denoise - post-processing(thier model) - HL compenstation -  AGC -> enhancer
    - lee: headin rotate/ se -> hearing loss compesntation -> amplifid speech 
    - cornell: dnn -> dnn -> beamformer -> dnn -> nalr + compression
    - tu: denoise(SNR loss, 1st)-> amplifier -> hearing loss model -> STOI/Loudness loss(2nd)

- STFT LSTM
    - 3 layer LSTM 1792, 2 stage, PIT, ReLU, PSM
    - [V] 3 layer BLSTM 896, 2 stage, PIT, ReLU, dropout 0.5
        - [STOP] MSE(ING): result/rnn-stft-mask/20230216-164227
        - PSM(nan)
    - [X, Not good] 3 layer BLSTM 896, 2 stage, no PIT, ReLU, 
    
- How to see several sources and channels?
    - Mix(batch, channels) -> (batch, sources * channels) -> Result(batch, sources, channels)

- Is PIT loss is too big???
    - STFT: Batch*Frame*Frequency
    - Wavform: Batch*nsample
    - Loss surface

### Experiments factor
- segment 1 -> 4: loss in each step has a convergence, especially validation also has.
- fixed buggy PIT algorithm(-> verify: choose nspk -> all of srcs are same)
    - loss in each step has a convergence
- channel 1 <-> 0, 1, 2, 3
- target channel: target vs anarchoic
- deverberation model by targeting anarchoic vs clean -> no time, stopping

### Experiments result
- Introduction
    - Clean vs Deverb amplified signal - haspi / hasqi

- Result 
    - Mixture vs Denoiser 
    - Mixture amplified vs Denoiser amplified vs Deverb in amplified

- HASPI, HASQI
    1. Unprocessed, NAL-R, Processed
    2. with finetune and without finetue in NALR
