# DACS

# Environment
    - transformers==4.17.0
    - datasets
    - jiwer
    - Levenshtein==0.21.0

# Training
1. Baselines
    - Fine-tune: use `finetune_ASRs.py` to train
        <details><summary>Show important arguments</summary>

        - `--model_type`: choose from wav2vec, data2vec, hubert, sewd, and unispeech

        </details>

        - The resulted model is then used for models **except GRL**
    - GRL: use `trainer_data2vec.py` to train
        <details><summary>Show important arguments</summary>

        - `--LAMBDA`: config for GRL, 0.5 as default
        - `--GRL`: once given in the command, the code will perform GRL training, o.w. multi-task will be performed
        - `--model_in_path`: path for the starting model, pre-trained model is used here
        - `--model_out_path`: path to save the resulted model
        - `--log_path`: path to save log file

        </details>

        - ASR decoder and AD classifier are added after the ASR encoder. GRL is added on the AD classifier.
    - Single Toggling: use `trainer_data2vec_toggle.py` to train
        <details><summary>Show important arguments</summary>

        - `--STAGE`: number of stage for training

        </details>

        ![images](https://github.com/Victoria-Wei/dementia-attribute-cancellation-strategy/blob/main/imgs/single_toggling.png)
        - Stage 1: freeze fine-tuned ASR encoder and train AD classifier
        - Stage 2: train toggling network with only **1 branch**(toggling network only generates vector with dim=2*D, where D is the dim. for ASR embedding. Only AD-free ASR score exists and is turned into mask by passing gumbel_softmax. Trained with L_ctc and reversed CE loss of AD classifier
    - FSM: use `trainer_data2vec_5st.py` to train
        - Stage 1: freeze fine-tuned ASR encoder and train AD classifier (use other code like `trainer_data2vec_toggle.py` to obtain)
        - Stage 2 (6 in the code): train 2 FSM at the same time
            - threshold is set to 0.5 to generate mask
2. Proposed
    - DACS: use `trainer_data2vec_2st.py` to train
        <details><summary>Show important arguments</summary>

        - `--AD_loss`: type of loss for AD classifier, can be chosen from the following types: cel, f1, recall, prec, (recall_ori, prec_ori)
        - `--checkpoint`: path to checkpoint so that training from checkpoint is possible
        - `--TOGGLE_RATIO`: for exp. to change toggle rate, y0' = (y1-y0)*TOGGLE_RATIO + y0
        - `--GS_TAU`: temperature for gumbel_softmax
        - `--W_LOSS`: weight for HC and AD

        </details>

        - Stage 1: freeze fine-tuned ASR encoder and train AD classifier, e.g.`python trainer_data2vec_2st.py -lam 0.5 -st 1 --AD_loss "recall" --W_LOSS 0.8 0.2 -model_in "./saves/data2vec-audio-large-960h_finetuned/final/" -model_out "./saves/data2vec-audio-large-960h_new1_recall_82" -log "data2vec-audio-large-960h_new1_recall_82.txt"`
        - Stage 2: train toggling network from stage 1 model, e.g. `python trainer_data2vec_2st.py -lam 0.5 -st 2 --AD_loss "recall" --W_LOSS 0.8 0.2 -model_in "./saves/data2vec-audio-large-960h_new1_recall_82/final/" -model_out "./saves/data2vec-audio-large-960h_new2_recall_82" -log "data2vec-audio-large-960h_new2_recall_82.txt"`


# Extracting Feat.
1. Baselines
    - Fine-tune: use `eval_finetune.py` to extract embeddings
        <details><summary>Show important arguments</summary>

        - `--model_path`: path to the model you want to extract
        - `--csv_path`: name for the csv file

        </details>
    - GRL: use `eval.py` to extract embeddings
        - Here we use stage=1 to represent GRL
    - Single Toggling: use `eval_SingleToggle.py` to extract embeddings (paired with `trainer_data2vec_toggle.py` with similar arguments)
    - FSM: use `eval_FSM.py`  to extract embeddings (paired with `trainer_data2vec_5st.py` with similar arguments)

2. Proposed
    - DACS: use `eval_toggle_GS.py`  to extract embeddings (paired with `trainer_data2vec_2st.py` with similar arguments)

3. Exp.
    - `eval_toggle_more.py` is used for all 3 exp. that force to toggle on more or less.

        We pass score y0 and y1 into gumbel_softmax mechanism to determine whether to toggle on certain node, and the larger the value of y0-y1 is, the more possible the decision will be to toggle on. In the later exp., we use y0-y1 to decide whether to toggle on or not.
        <details><summary>Show important arguments</summary>

        - `--exp_type`: type of exp., can be chosen from the following types: `h` for homogeneous masking, `a` for aggressive masking, and `p` for passive masking
        - `--NUM_OFF`: num of groups to toggle off for homogeneous masking
        - `--AP_RATIO`: ratio for aggressive & passive masking
        </details>

        - Homogeneous masking: toggle off `NUM_OFF` groups of nodes with smallest y0-y1, `NUM_OFF` from 0 to 15, e.g. `python eval_toggle_more.py -st 2 -model_type "data2vec" --AD_loss "recall" --exp_type "h" --NUM_OFF 3 -model "./saves/data2vec-audio-large-960h_new2_recall/final/" -csv "data2vec-audio-large-960h_new2_recall_3off"`
        - Aggressive masking: toggle off more. Those originally toggled off are still off,  `AP_RATIO` of the nodes that were toggled on will be toggled off according to their y0-y1 values. The smaller the more likely to be toggled off. e.g. `python eval_toggle_more.py -st 2 -model_type "data2vec" --AD_loss "recall" --exp_type "a" --AP_RATIO 0.8 -model "./saves/data2vec-audio-large-960h_new2_recall/final/" -csv "data2vec-audio-large-960h_new2_recall_a80"`
        - Passive masking: toggle on more. Those originally toggled on are still on. `AP_RATIO` of the nodes that were toggled off will be toggled on according to their y0-y1 values. The larger the more likely to be toggled on. e.g `python eval_toggle_more.py -st 2 -model_type "data2vec" --AD_loss "recall" --exp_type "p" --AP_RATIO 0.2 -model "./saves/data2vec-audio-large-960h_new2_recall/final/" -csv "data2vec-audio-large-960h_new2_recall_p20"`
    
# Evaluation
1. ASR performance in WER  
use `-v` to set how many groups you want, `-csv` to set the path to xetracted feats. from last section, and `-save` to set path to the folder where to want to keep WER files, e.g.  
`python detail_wer.py -v 3 -csv "./saves/results/data2vec-audio-large-960h_new2_recall_ori.csv" -save "./saves/results/detail_wer/data2vec-audio-large-960h_new2_recall_ori" -T`
2. SVM AD Prediction  
use `-model` to set model name and `-sq` to set how to squeeze embeddings (min/max/mean), e.g.  
`python pred_AD_svm.py -model "data2vec-audio-large-960h_new2_prec" -sq 'min'`
3. Masks & Masked Embeddings' Characteristic
    - First use `feat_scoring.py` to extract
        - node-based: [MEX_rates, MIs, lm_node_on_rate, AD_node_on_rate, rates_11] saved in `"./saves/results/FSM_info/" + args.model_name + '.csv'`
            - MEX_rates: rate of "mutually exclusion" along time-axis
            - MIs: mutual info along time-axis
            - lm_node_on_rate: rate of ASR-node turning on along time-axis
            - AD_node_on_rate: rate of AD-node turning on along time-axis
            - rates_11: rate of 2 masks being 1s along time-axis
        - utt-based: [lm_on_rate, AD_on_rate] saved in `"./saves/results/FSM_info/" + args.model_name + '_onRate.csv'`
            - lm_on_rate: for each time-step, rate of lm_mask being 1s, average along time-axis
            - AD_on_rate: for each time-step, rate of AD_mask being 1s, average along time-axis
        - node-wise: similar to node-based w.o. averaging over utt
            - MEX_rates_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_MEX_rates.csv'`
            - MIs_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_MIs.csv'`
            - lm_node_on_rate_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_lm_node_on_rate.csv'`
            - AD_node_on_rate_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_AD_node_on_rate.csv'`
            - rates_11_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_rates_11.csv'`
        - Masked Emb.
            - Print average of masked embeddings of AD- and ASR-masked
            - Masked Emb.s saved in `"./saves/results/FSM_info/" + args.model_name + '_lm_masked_embs.csv'`, `"./saves/results/FSM_info/" + args.model_name + '_ad_masked_embs.csv'`, and `"./saves/results/FSM_info/" + args.model_name + '_un_masked_embs.csv'`
    - Use `mask_info.ipynb` to visualize
