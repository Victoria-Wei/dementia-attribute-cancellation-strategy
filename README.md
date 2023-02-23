# DACS

# Training
1. Baselines
    - Fine-tune
    - GRL
    - Single Toggling
    - FSM
2. Proposed
    - DACS

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
