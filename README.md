# DACS

# Training
1. Baselines
    - Fine-tune
    - GRL
    - Single Toggling
    - FSM
2. Proposed
    - DACS: train for 2 stage. 
        <details><summary>Show important arguments</summary>

        - `--AD_loss`: type of loss for AD classifier, can be chosen from the following types: cel, f1, recall, prec, (recall_ori, prec_ori)
        - `--checkpoint`: path to checkpoint so that training from checkpoint is possible
        - `--TOGGLE_RATIO`: for exp. to change toggle rate, y0' = (y1-y0)*TOGGLE_RATIO + y0
        - `--GS_TAU`: temperature for gumbel_softmax
        - `--W_LOSS`: weight for HC and AD
        </details>

        - Stage 1: train AD classifier from fine-tune model, e.g.`python trainer_data2vec_2st.py -lam 0.5 -st 1 --AD_loss "recall" --W_LOSS 0.8 0.2 -model_in "./saves/data2vec-audio-large-960h_finetuned/final/" -model_out "./saves/data2vec-audio-large-960h_new1_recall_82" -log "data2vec-audio-large-960h_new1_recall_82.txt"`
        - Stage 2: train toggling network from stage 1 model, e.g. `python trainer_data2vec_2st.py -lam 0.5 -st 2 --AD_loss "recall" --W_LOSS 0.8 0.2 -model_in "./saves/data2vec-audio-large-960h_new1_recall_82/final/" -model_out "./saves/data2vec-audio-large-960h_new2_recall_82" -log "data2vec-audio-large-960h_new2_recall_82.txt"`


# Extracting Feat.
1. Baselines
    - Fine-tune
    - GRL
    - Single Toggling
    - FSM
2. Proposed
    - DACS
3. Exp.
    - eval_toggle_more.py
    
# Evaluation
1. ASR performance in WER  
use `-v` to set how many groups you want, `-csv` to set the path to xetracted feats. from last section, and `-save` to set path to the folder where to want to keep WER files, e.g.  
`python detail_wer.py -v 3 -csv "./saves/results/data2vec-audio-large-960h_new2_recall_ori.csv" -save "./saves/results/detail_wer/data2vec-audio-large-960h_new2_recall_ori" -T`
2. SVM AD Prediction  
use `-model` to set model name and `-sq` to set how to squeeze embeddings (min/max/mean), e.g.  
`python pred_AD_svm.py -model "data2vec-audio-large-960h_new2_prec" -sq 'min'`
3. Masks & Masked Embeddings' Characteristic
    - First use `feat_scoring.py` to extract
        - node-based (針對每個node去計算、平均over utt，最後會有hidden_size個值?): [MEX_rates, MIs, lm_node_on_rate, AD_node_on_rate, rates_11] saved in `"./saves/results/FSM_info/" + args.model_name + '.csv'`
            - MEX_rates: rate of "mutually exclusion" along time-axis(在時間上，ASR-mask跟AD-mask不同的比例)
            - MIs: mutual info along time-axis(在時間上，ASR-mask跟AD-mask的mutual info)
            - lm_node_on_rate: rate of ASR-node turning on along time-axis(在時間上，ASR-mask開的比例)
            - AD_node_on_rate: rate of AD-node turning on along time-axis(在時間上，AD-mask開的比例)
            - rates_11: rate of 2 masks being 1s along time-axis(在時間上，ASR-mask跟AD-mask皆為1的比例)
        - utt-based (針對每個音檔去計算，最後會有num_utt個值?): [lm_on_rate, AD_on_rate] saved in `"./saves/results/FSM_info/" + args.model_name + '_onRate.csv'`
            - lm_on_rate: 每個time-step，這條mask開的比例，平均over整個音檔
            - AD_on_rate: 每個time-step，這條mask開的比例，平均over整個音檔
        - node-wise (針對每個node、每個utt去計算，最後會有hidden_size*num_utt個值?): similar to node-based w.o. averaging over utt
            - MEX_rates_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_MEX_rates.csv'`
            - MIs_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_MIs.csv'`
            - lm_node_on_rate_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_lm_node_on_rate.csv'`
            - AD_node_on_rate_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_AD_node_on_rate.csv'`
            - rates_11_df saved in `"./saves/results/FSM_info/" + args.model_name + '_NodeWise_rates_11.csv'`
        - Masked Emb.
            - Print average of masked embeddings of AD- and ASR-masked
            - Masked Emb.s saved in `"./saves/results/FSM_info/" + args.model_name + '_lm_masked_embs.csv'`, `"./saves/results/FSM_info/" + args.model_name + '_ad_masked_embs.csv'`, and `"./saves/results/FSM_info/" + args.model_name + '_un_masked_embs.csv'`
    - Use `mask_info.ipynb` to visualize
