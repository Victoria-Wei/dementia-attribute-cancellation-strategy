import pandas as pd
from jiwer.transformations import wer_default, wer_standardize, cer_default_transform
from jiwer import transforms as tr
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

def _preprocess(
    truth: List[str],
    hypothesis: List[str],
    truth_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
) -> Tuple[List[str], List[str]]:
    """
    Pre-process the truth and hypothesis into a form such that the Levenshtein
    library can compute the edit operations.can handle.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: the preprocessed truth and hypothesis
    """
    # Apply transforms. The transforms should collapses input to a list of list of words
    transformed_truth = truth_transform(truth)
    transformed_hypothesis = hypothesis_transform(hypothesis)

    # raise an error if the ground truth is empty or the output
    # is not a list of list of strings
    if len(transformed_truth) != len(transformed_hypothesis):
        raise ValueError(
            "number of ground truth inputs ({}) and hypothesis inputs ({}) must match.".format(
                len(transformed_truth), len(transformed_hypothesis)
            )
        )
    if not _is_list_of_list_of_strings(transformed_truth, require_non_empty_lists=True):
        raise ValueError(
            "truth should be a list of list of strings after transform which are non-empty"
        )
    if not _is_list_of_list_of_strings(
        transformed_hypothesis, require_non_empty_lists=False
    ):
        raise ValueError(
            "hypothesis should be a list of list of strings after transform"
        )

    # tokenize each word into an integer
    vocabulary = set(chain(*transformed_truth, *transformed_hypothesis))

    if "" in vocabulary:
        raise ValueError(
            "Empty strings cannot be a word. "
            "Please ensure that the given transform removes empty strings."
        )

    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    truth_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in transformed_truth
    ]
    hypothesis_chars = [
        "".join([chr(word2char[w]) for w in sentence])
        for sentence in transformed_hypothesis
    ]

    return truth_chars, hypothesis_chars


def _is_list_of_list_of_strings(x: Any, require_non_empty_lists: bool):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if require_non_empty_lists and len(e) == 0:
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True


def compute_measures(
    src: Union[str, List[str]],
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    save_path,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = wer_default,
    **kwargs
) -> Dict[str, float]:
    """
    Calculate error measures between a set of ground-truth sentences and a set of
    hypothesis sentences.
    The set of sentences can be given as a string or a list of strings. A string
    input is assumed to be a single sentence. A list of strings is assumed to be
    multiple sentences which need to be evaluated independently. Each word in a
    sentence is separated by one or more spaces. A sentence is not expected to end
    with a specific token (such as a `.`). If the ASR system does delimit sentences
    it is expected that these tokens are filtered out.
    The optional `transforms` arguments can be used to apply pre-processing to
    respectively the ground truth and hypotheses input. By default, the following
    transform is applied to both the ground truth and hypothesis string(s). These
    steps are required and necessary in order to compute the measures.
    1) The start and end of a string are stripped of white-space symbols
    2) Contiguous spaces (e.g `   `) are reduced to a single space (e.g ` `)
    3) A sentence (with a single space (` `) between words) is reduced to a
       list of words
    Any non-default transformation is required to reduce the input to at least
    one list of words in order to facility the computation of the edit distance.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: a dict with WER, MER, WIP and WIL measures as floating point numbers
    """
    # deprecated old API
    if "standardize" in kwargs:
        warnings.warn(
            UserWarning(
                "keyword argument `standardize` is deprecated. "
                "Please use `truth_transform=jiwer.transformations.wer_standardize` and"
                " `hypothesis_transform=jiwer.transformations.wer_standardize` instead"
            )
        )
        truth_transform = wer_standardize
        hypothesis_transform = wer_standardize
    if "words_to_filter" in kwargs:
        warnings.warn(
            UserWarning(
                "keyword argument `words_to_filter` is deprecated. "
                "Please compose your own transform with `jiwer.transforms.RemoveSpecificWords"
            )
        )
        t = tr.RemoveSpecificWords(kwargs["words_to_filter"])
        truth = t(truth)
        hypothesis = t(hypothesis)

    # validate input type
    if isinstance(truth, str):
        truth = [truth]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in truth):
        raise ValueError("one or more groundtruths are empty strings")

    # Preprocess truth and hypothesis
    trans = truth
    pred = hypothesis

    truth, hypothesis = _preprocess(
        truth, hypothesis, truth_transform, hypothesis_transform
    )

    # keep track of total hits, substitutions, deletions and insertions
    # across all input sentences
    H, S, D, I = 0, 0, 0, 0

    # also keep track of the total number of ground truth words and hypothesis words
    gt_tokens, hp_tokens = 0, 0
    
    i = 0
    for groundtruth_sentence, hypothesis_sentence in zip(truth, hypothesis):
        # Get the operation counts (#hits, #substitutions, #deletions, #insertions)
        with open(save_path, 'a') as f:
            f.write(str(src[i]) + ": (ref) " + str(trans[i]) + "\n")       # current file info
            f.write(str(src[i]) + ": (hyp) " + str(pred[i]) + "\n")
            f.write("SDI in order:")
            
        hits, substitutions, deletions, insertions = _get_operation_counts(
            save_path,
            groundtruth_sentence, hypothesis_sentence
        )
        with open(save_path, 'a') as f:
            f.write("\nC: " + str(hits) + "; S: " + str(substitutions) + "; D: " + str(deletions) + "; I: " + str(insertions) + "\n")
            f.write("====================\n")
            
        H += hits
        S += substitutions
        D += deletions
        I += insertions
        gt_tokens += len(groundtruth_sentence)
        hp_tokens += len(hypothesis_sentence)
        i = i + 1

    # Compute Word Error Rate
    wer = float(S + D + I) / float(H + S + D)

    # Compute Match Error Rate
    mer = float(S + D + I) / float(H + S + D + I)

    # Compute Word Information Preserved
    wip = (float(H) / gt_tokens) * (float(H) / hp_tokens) if hp_tokens >= 1 else 0

    # Compute Word Information Lost
    wil = 1 - wip
    
    with open(save_path, 'a') as f:
        f.write("Overall Results:\n")
        f.write("wer: " + str(wer) + "; hits: " + str(H) + "; substitutions: " + str(S) + "; deletions: " + str(D) + "; insertions: " + str(I))

    return {
        "wer": wer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "hits": H,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
    }
import Levenshtein
def _get_operation_counts(
    save_path,
    source_string: str, destination_string: str
) -> Tuple[int, int, int, int]:
    """
    Check how many edit operations (delete, insert, replace) are required to
    transform the source string into the destination string. The number of hits
    can be given by subtracting the number of deletes and substitutions from the
    total length of the source string.
    :param source_string: the source string to transform into the destination string
    :param destination_string: the destination to transform the source string into
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """
    editops = Levenshtein.editops(source_string, destination_string)
    for op in editops:
        with open(save_path, 'a') as f:
            f.write(str(op[0]) + " ")
            
    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)
    
    return hits, substitutions, deletions, insertions

import numpy as np
def ID2MMSE(ID,
            id2mmse = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/ID2MMSE_train.npy", allow_pickle=True).tolist()):

    name = ID.split("_")                                  # from file name to spkID
    if (name[1] == 'INV'):                                # shouldn't have INV
        MMSE = None
    else:                                                 # for participant
        MMSE = id2mmse[name[0]]                           # label according to look-up table
    return MMSE                                           # return MMSE for this file

import os
def detail_wer(csv_path, detail_path, level, TEST):
    df = pd.read_csv(csv_path)                      # all data
    true_trans_lst = df.text.values.tolist()        # true trans
    df.pred_str.fillna(value="", inplace=True)      # replace nan in pred w/ ""
    pred_trans_lst = df.pred_str.values.tolist()    # pred trans
    utt_lst = df.path.values.tolist()               # file names
    AD_lst = df.dementia_labels.values.tolist()     # dementia labels
    
    if level >= 1:                                      # output overall detail
        if (os.path.exists(detail_path + "/overall.txt")):
            print("overall.txt exists.")
            return
        compute_measures(src=utt_lst, truth=true_trans_lst, hypothesis=pred_trans_lst, save_path=detail_path + "/overall.txt")
    if level >= 2:                                      # separate AD & HC details
        if (os.path.exists(detail_path + "/HC.txt")):
            print("HC.txt exists.")
            return 
        if (os.path.exists(detail_path + "/AD_all.txt")):
            print("AD_all.txt exists.")
            return
        
        sub = df.loc[(df.dementia_labels == 0),]        # divide data into sub-data
        print("# of HC: ", len(sub))
        true_trans = sub.text.values.tolist()           # true trans of sub-data
        pred_trans = sub.pred_str.values.tolist()       # pred trans of sub-data
        utt = sub.path.values.tolist()                  # file names of sub-data
        
        compute_measures(src=utt, truth=true_trans, hypothesis=pred_trans, save_path=detail_path + "/HC_all.txt")
        
        sub1 = sub.loc[(sub.path.str.contains('INV')),] # for interveiwer only
        print("# of HC of INV: ", len(sub1))
        true_trans = sub1.text.values.tolist()          # true trans of sub-data
        pred_trans = sub1.pred_str.values.tolist()      # pred trans of sub-data
        utt = sub1.path.values.tolist()                 # file names of sub-data
        
        compute_measures(src=utt, truth=true_trans, hypothesis=pred_trans, save_path=detail_path + "/HC_INV.txt")
        
        sub1 = sub.loc[(sub.path.str.contains('PAR')),] # for participant only
        print("# of HC of PAR: ", len(sub1))
        true_trans = sub1.text.values.tolist()          # true trans of sub-data
        pred_trans = sub1.pred_str.values.tolist()      # pred trans of sub-data
        utt = sub1.path.values.tolist()                 # file names of sub-data
        
        compute_measures(src=utt, truth=true_trans, hypothesis=pred_trans, save_path=detail_path + "/HC_PAR.txt")
        
        sub = df.loc[(df.dementia_labels == 1),]        # divide data into sub-data
        print("# of AD: ", len(sub))
        true_trans = sub.text.values.tolist()           # true trans of sub-data
        pred_trans = sub.pred_str.values.tolist()       # pred trans of sub-data
        utt = sub.path.values.tolist()                  # file names of sub-data
        
        compute_measures(src=utt, truth=true_trans, hypothesis=pred_trans, save_path=detail_path + "/AD_all.txt")
    if level >= 3:                                      # further separate AD samples
        AD_bounds = [30, 24, 20, 9, -1]
        if TEST:
            id2mmse = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/ID2MMSE.npy", allow_pickle=True).tolist()
        else:
            id2mmse = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/ID2MMSE_train.npy", allow_pickle=True).tolist()
       
        for i in range(4):                              # 4 level: normal cognition, mild, moderate, and severe
            j = 0
            print("Current range: ", AD_bounds[i+1]+1, "-", AD_bounds[i])
            txt_name = "/AD_" + str(AD_bounds[i+1]+1) + "-" + str(AD_bounds[i]) + ".txt"
                                                        # get txt name
            if (os.path.exists(detail_path + txt_name)):
                print(txt_name + " exists.")
                return
        
            sub_utt = []
            true_txt = []
            pred_txt = []
            for k in range(len(sub)):                   # each AD sample
                ID = utt[k]                             # get ID
                if ID2MMSE(ID, id2mmse) != None:
                    mmse = int(ID2MMSE(ID, id2mmse))    # get mmse score as int
                    if mmse <= AD_bounds[i] and mmse > AD_bounds[i+1]:
                                                        # in current range
                            sub_utt.append(ID)
                            true_txt.append(true_trans[k])
                            pred_txt.append(pred_trans[k])
                            j = j + 1
            print("Total ", str(j), " samples.")
            if j !=0:
                compute_measures(src=sub_utt, truth=true_txt, hypothesis=pred_txt, save_path=detail_path + txt_name)

import argparse
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, default=1, help="1: all, 2: HC & AD, 3: HC & 4 levels of AD")
    parser.add_argument('-csv', '--csv_path', type=str, default="./saves/results/wav2vec2-base-960h_GRL_0.3_dev.csv", help="path of stored data, ex: ./saves/results/wav2vec2-base-960h_GRL_0.3_dev.csv")
    parser.add_argument('-save', '--save_dir', type=str, default="./saves/results/detail_wer/wav2vec2-base-960h_GRL_0.3/", help="dir to store detail wer, ex: ./saves/results/detail_wer/wav2vec2-base-960h_GRL_0.3/")
    parser.add_argument('-T', '--TEST', action='store_true', default=False, help="flag for testing data")
    args = parser.parse_args()
    
    verbose = args.verbose
    path = args.csv_path
    detail = args.save_dir
    TEST = args.TEST

    detail_wer(path, detail, verbose, TEST)
    print("All done")
    
if __name__ == "__main__":
    main()

    
