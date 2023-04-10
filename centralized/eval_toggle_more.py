# only data2vec model is used!!!!!!!!!!
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import os

from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, 
    Data2VecAudioModel, Data2VecAudioPreTrainedModel,
    HubertModel, HubertPreTrainedModel,
    SEWDModel, SEWDPreTrainedModel,
    UniSpeechSatModel, UniSpeechSatPreTrainedModel
)
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer, Wav2Vec2Encoder, Wav2Vec2EncoderStableLayerNorm
from transformers.configuration_utils import PretrainedConfig

from transformers.file_utils import (
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    ModelOutput,
    PushToHubMixin,
    copy_func,
    is_offline_mode,
    replace_return_docstrings,
    is_datasets_available,
    add_code_sample_docstrings,
)

from transformers.modeling_utils import PreTrainedModel

from transformers.file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput, MaskedLMOutput, SequenceClassifierOutput
from transformers.utils import logging
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

from transformers.debug_utils import DebugOption
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from huggingface_hub import Repository

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)

from transformers.utils import logging

class ReverseLayerF(torch.autograd.Function):
    def __init__(self):
        super(ReverseLayerF, self).__init__()
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel
    
WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    Args:
        input_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the :class:`~transformers.Wav2Vec2Processor` should
            be used for padding and conversion into a tensor of type `torch.FloatTensor`. See
            :meth:`transformers.Wav2Vec2Processor.__call__` for details.
        attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in ``[0,
            1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
            .. warning::
                :obj:`attention_mask` should only be passed if the corresponding processor has
                ``config.return_attention_mask == True``. For all models whose processor has
                ``config.return_attention_mask == False``, such as `wav2vec2-base
                <https://huggingface.co/facebook/wav2vec2-base-960h>`__, :obj:`attention_mask` should **not** be passed
                to avoid degraded performance when doing batched inference. For such models :obj:`input_values` should
                simply be padded with 0 and passed without :obj:`attention_mask`. Be aware that these models also yield
                slightly different results depending on whether :obj:`input_values` is padded or not.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
        
DATA2VEC_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            <Tip warning={true}>
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [data2vec-audio-base](https://huggingface.co/facebook/data2vec-audio-base-960h), `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.
            </Tip>
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

HUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            <Tip warning={true}>
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [hubert-base](https://huggingface.co/facebook/hubert-base-ls960), `attention_mask` should **not** be passed
            to avoid degraded performance when doing batched inference. For such models `input_values` should simply be
            padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly different
            results depending on whether `input_values` is padded or not.
            </Tip>
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

SEWD_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`Wav2Vec2Processor`] should be used for padding
            and conversion into a tensor of type *torch.FloatTensor*. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

UNISPEECH_SAT_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`UniSpeechSatProcessor`] should be used for
            padding and conversion into a tensor of type *torch.FloatTensor*. See [`UniSpeechSatProcessor.__call__`]
            for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            <Tip warning={true}>
            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, such as
            [unispeech_sat-base](https://huggingface.co/facebook/unispeech_sat-base-960h), `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.
            </Tip>
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

logger = logging.get_logger(__name__)

import pandas as pd
from datasets import Dataset
import librosa
from datasets import load_dataset, load_from_disk
from transformers import Wav2Vec2Processor
import torch
from jiwer import wer

def ID2Label(ID,
            spk2label = np.load("/mnt/Internal/FedASR/weitung/HuggingFace/Pretrain/dataset/test_dic.npy", allow_pickle=True).tolist()):
    name = ID.split("_")                                                    #  from file name to spkID
    if (name[1] == 'INV'):                                                  # interviewer is CC
        label = 0
    else:                                                                   # for participant
        label = spk2label[name[0]]                                          # label according to look-up table
    return label                                                            # return dementia label for this file

def csv2dataset(PATH = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/clips/',
                path = '/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv'):
    stored = "./dataset/" + path.split("/")[-1].split(".")[0]
    if (os.path.exists(stored)):
        print("Load data from local...")
        return load_from_disk(stored)
 
    data = pd.read_csv(path)                                                # read desired csv
    dataset = Dataset.from_pandas(data)                                     # turn into class dataset
    
    # initialize a dictionary
    my_dict = {}
    my_dict["path"] = []                                                    # path to audio
    my_dict["array"] = []                                                   # waveform in array
    my_dict["text"] = []                                                    # ground truth transcript
    my_dict["dementia_labels"] = []

    i = 1
    for path in dataset['path']:                                            # for all files
        if dataset['sentence'][i-1] != None:                                # only the non-empty transcript
            sig, s = librosa.load(PATH + path, sr=16000, dtype='float32')   # read audio w/ 16k sr
            if len(sig) > 1600:                                             # get rid of audio that's too short
                my_dict["path"].append(path)                                # add path
                my_dict["array"].append(sig)                                # add audio wave
                my_dict["text"].append(dataset['sentence'][i-1].upper())    # transcript to uppercase
                my_dict["dementia_labels"].append(ID2Label(path))
        print(i, end="\r")                                                  # print progress
        i += 1
    print("There're ", len(my_dict["path"]), " non-empty files.")

    result_dataset = Dataset.from_dict(my_dict)
    result_dataset.save_to_disk(stored)                                     # save for later use
    
    return result_dataset

def prepare_dataset(batch):
    audio = batch["array"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch


import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        AD_labels = [{"dementia_labels": feature["dementia_labels"]} for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",                                   # to torch tensor
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["dementia_labels"] = torch.tensor([torch.tensor(d['dementia_labels']) for d in AD_labels]) # list of dict to list of tensor
        return batch

from datasets import load_metric
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from functions.models import AngularPenaltySMLoss

def FSMatt_loss(lm_masks, dementia_masks):                       # calculate cos similarity for each sample avg over samples
    loss = 0
    for i in range(len(lm_masks)):                               # for each sample
        lm_mask = lm_masks[i]
        AD_mask = dementia_masks[i]

        lm_mask_mean = torch.mean(lm_mask,dim=0)                 # average along t-axis
        dementia_mask_mean = torch.mean(AD_mask,dim=0)           # average along t-axis

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        s12 = cos(lm_mask_mean, dementia_mask_mean)              # cosine similarity
        s21 = cos(dementia_mask_mean, lm_mask_mean)
        row1 = torch.cat((torch.Tensor([0]).to(s12.device), torch.Tensor([s12]).to(s12.device)), 0)
        row2 = torch.cat((torch.Tensor([s21]).to(s21.device), torch.Tensor([0]).to(s21.device)), 0)
        row1 = torch.unsqueeze(row1, 0)
        row2 = torch.unsqueeze(row2, 0)
        S = torch.cat((row1, row2), 0)                           # [[0, s12], [s21, 0]]
        loss += torch.norm(S, p='fro')                           # Frobenius norm
    return loss / (i+1)                                          # average over samples

##################################
# choose model type
##################################    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current stage")
parser.add_argument('-model', '--model_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5/checkpoint-14010/", help="Where the model is saved")
parser.add_argument('-csv', '--csv_path', type=str, default="wav2vec2-base-960h_GRL_0.5", help="name for the csv file")
parser.add_argument('-thres', '--threshold', type=float, default=0.5, help="Threshold for AD & ASR")
parser.add_argument('-model_type', '--model_type', type=str, default="wav2vec2", help="Type of the model")
# 2023/01/08: loss type
parser.add_argument('-ad_loss', '--AD_loss', type=str, default="cel", help="loss to use for AD classifier")
# 2023/02/13: toggling ratio
parser.add_argument('-toggle_rt', '--TOGGLE_RATIO', type=float, default=0, help="To toggle more or less")
# 2023/02/15: TOGGLE_RATIO, loss weight
parser.add_argument('-gs_tau', '--GS_TAU', type=float, default=1, help="Tau for gumbel_softmax")
parser.add_argument('-w_loss', '--W_LOSS', type=float, default=None, nargs='+', help="weight for HC and AD")
# 2023/02/23: exp. for toggling more
parser.add_argument('-exp', '--exp_type', type=str, default=None, help="Type of the experiment: homogeneous masking(h), aggressive masking(a), and passive masking(p)")
parser.add_argument('-num_off', '--NUM_OFF', type=int, default=None, help="num of groups to set off")
# 2023/03/07: ratio for aggressive & passive masking
parser.add_argument('-ap_rt', '--AP_RATIO', type=float, default=0, help="To toggle more or less for aggressive & passive masking")

args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_dir = args.model_path             # path to load the model
csv_name = args.csv_path                # path to store the result
model_type = args.model_type            # select different type of model (here only data2vec is ready to use)
AD_loss = args.AD_loss                  # type of AD loss: cel, f1, recall, prec, (recall_ori, prec_ori)
TOGGLE_RATIO = args.TOGGLE_RATIO        # for exp. to change toggle rate
GS_TAU = args.GS_TAU                    # temperature for gumbel_softmax
if args.W_LOSS == None:                 # weight for HC and AD
    W_LOSS = [0.1, 0.9]                 # default weight for HC and AD
else:
    W_LOSS = args.W_LOSS
print("weight for loss: ", W_LOSS)
EXP_TYPE = args.exp_type                # type of exp. that "forces to toggle more or less": homogeneous masking(h), aggressive masking(a), and passive masking(p)
NUM_OFF = args.NUM_OFF                  # num of groups to toggle off for homogeneous masking
AP_RATIO = args.AP_RATIO                # ratio for aggressive & passive masking
print("Current exp.: ", EXP_TYPE, " with ", NUM_OFF, "groups off.", " and AP_RATIO=", AP_RATIO)

# threshold for maskes
AD_THRES = args.threshold
LM_THRES = args.threshold

# model type
if model_type == "wav2vec":
    print("Now, using wav2vec model")
    _CONFIG_FOR_DOC = "Wav2Vec2Config"

    _PROCESSOR_FOR_DOC = None
    _CHECKPOINT_FOR_DOC = None
    _CTC_EXPECTED_OUTPUT = None
    _CTC_EXPECTED_LOSS = None

elif model_type == "data2vec":
    print("Now, using data2vec model")
    _CONFIG_FOR_DOC = "Data2VecAudioConfig"
    _PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
    _CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"
    _CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
    _CTC_EXPECTED_LOSS = 66.95
elif model_type == "hubert":
    print("Now, using hubert model")
    _CONFIG_FOR_DOC = "HubertConfig"
    _PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
    _CHECKPOINT_FOR_DOC = "facebook/hubert-large-ls960-ft"
    _CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
    _CTC_EXPECTED_LOSS = 22.68
elif model_type == "sewd":
    print("Now, using sewd model")
    _CONFIG_FOR_DOC = "SEWDConfig"
    _PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
    _CHECKPOINT_FOR_DOC = "asapp/sew-d-tiny-100k-ft-ls100h"
    _CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTIL OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
    _CTC_EXPECTED_LOSS = 0.21
elif model_type == "unispeech":
    print("Now, using unispeech model")
    _CONFIG_FOR_DOC = "UniSpeechSatConfig"
    _PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
    _CHECKPOINT_FOR_DOC = "microsoft/unispeech-sat-base-100h-libri-ft"
    _CTC_EXPECTED_OUTPUT = "'MISTER QUILDER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
    _CTC_EXPECTED_LOSS = 39.88

class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`."
                "or define `vocab_size` of your model's configuration."
            )
        
        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        #self.lm_fsm = nn.LSTM(config.hidden_size, config.hidden_size, 1)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_extractor()
        
        if STAGE == 1:                                                  # train FSM
            #print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            #print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train encoder
            """
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            """

            # train lm_FSM
            
            self.freeze_wav2vec2()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            
            
            # train dementia_FSM
            """
            self.freeze_wav2vec2()
            self.freeze_lm_fsm()          
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            """
            
        self.init_weights()
    
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor so that its parameter
        will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_wav2vec2(self):
        self.wav2vec2.eval()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
            
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
            
    def freeze_dementia_fsm(self):
        self.dementia_fsm.eval()
        for param in self.dementia_fsm.parameters():
            param.requires_grad = False
            
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
   
    def freeze_lm_grl(self):
        self.lm_grl.eval()
        for param in self.lm_grl.parameters():
            param.requires_grad = False
 
    def freeze_dementia_grl(self):
        self.dementia_grl.eval()
        for param in self.dementia_grl.parameters():
            param.requires_grad = False
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_length)`, `optional`):
            Labels for connectionist temporal classification. Note that ``target_length`` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in ``[-100, 0, ..., config.vocab_size -
            1]``. All labels set to ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ...,
            config.vocab_size - 1]``.
        Returns:
        Example::
            >>> import torch
            >>> from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            >>> from datasets import load_dataset
            >>> import soundfile as sf
            >>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            >>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            >>> def map_to_array(batch):
            >>>     speech, _ = sf.read(batch["file"])
            >>>     batch["speech"] = speech
            >>>     return batch
            >>> ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            >>> ds = ds.map(map_to_array)
            >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
            >>> logits = model(input_values).logits
            >>> predicted_ids = torch.argmax(logits, dim=-1)
            >>> transcription = processor.decode(predicted_ids[0])
            >>> # compute loss
            >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"
            >>> # wrap processor as target processor to encode labels
            >>> with processor.as_target_processor():
            >>>     labels = processor(target_transcription, return_tensors="pt").input_ids
            >>> loss = model(input_values, labels=labels).loss
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        
        # hidden_states: wav2vec embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))             # score range from 0~1
        
        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # loss for lm_grl
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)

                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if STAGE == 1:                                                  # train FSM
                    #print("Current stage: 1")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 2:                                                # train ASR
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 3:                                                # train dementia GRL
                    #print("Current stage: 3")
                    final_loss = dementia_loss_rev
                elif STAGE == 4:
                    final_loss = loss_r
                elif STAGE == 5:
                    # train encoder
                    #final_loss = loss + dementia_loss + score_loss + Att_loss + dementia_loss_rev + loss_r
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                    # train dementia_FSM
                    #final_loss = dementia_loss + loss_r
                # ////
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'dementia_mask': dementia_mask, 'lm_mask': lm_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class RecallLoss(nn.Module):
    """ An unofficial implementation of
        <Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        recall = TP / (TP + FN)
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(RecallLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, input, target, AD_loss):
        input = input.to(torch.float)
        target = target.to(torch.int64)

        N, C = input.size()[:2]                                                         # [batch_size, 2]
        logpt = F.log_softmax(input, dim=1)
        pt = logpt.exp()                                                                # pred_prob: [batch_size, 2]
        #print("pt: ", pt)

        ## convert target (N, 1, *) into one hot vector (N, C, *)
        target = target.view(N, 1, -1)                                                  # (N, 1, *)
        last_size = target.size(-1)
        target_onehot = torch.zeros((N, C, last_size)).type_as(pt)                      # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)                                            # (N, C, *)

        true_positive = torch.sum(pt.view(N, C, last_size) * target_onehot, dim=2)      # (N, C): true label的預測"機率"
        total_target = torch.sum(target_onehot, dim=2)                                  # (N, C): true_prob

        ## Recall = TP / (TP + FN)
        recall = (true_positive + self.smooth) / (total_target + self.smooth)           # (N, C): true label的預測"機率", false label為1
        # --> 目標把"各個class"對的抓出來
        total_predict = torch.sum(pt.view(N, C, last_size), dim=2)                      # (N, C): pred_prob for all labels
        precision = (true_positive + 1e-5) / (total_predict + 1e-5)                     # (N, C): true label為1，false label為機率微擾後的倒數
        #  --> 目標false label的機率越小越好
        f1 = 2 * recall * precision / (recall + precision)


        if hasattr(self, 'weight'):
            if self.weight.type() != input.type():
                self.weight = self.weight.type_as(input)
            #print("weight: ", self.weight)
            recall_ori = recall * self.weight * C                                       # (N, C): recall
            precision_ori = precision * self.weight * C                                 # (N, C): prec
            f1 = f1 * self.weight * C                                                           # (N, C): f1
            recall = (torch.ones((N, C)).type_as(recall) - recall) * self.weight * C            # (N, C): 1 - recall
            precision = (torch.ones((N, C)).type_as(precision) - precision) * self.weight * C   # (N, C): 1 - prec

        #print("recall: ", recall)
        recall_loss = torch.mean(recall)  # mean越小越好，recall越小越好，1 - true label的預測"機率"越小越好 --> true label的預測"機率"越大越好
        prec_loss = torch.mean(precision) # mean越小越好，precision越小越好，{1 - false label的機率微擾後的倒數} 越小越好 --> false label的機率越小越好
        f1_loss = 1 - torch.mean(f1)
        recall_ori_loss = 1 - torch.mean(recall_ori)
        precision_ori_loss = 1 - torch.mean(precision_ori)

        if AD_loss == "f1":
            return f1_loss
        elif AD_loss == "recall":
            return recall_loss
        elif AD_loss == "prec":
            return prec_loss
        elif AD_loss == "recall_ori":
            return recall_ori_loss
        elif AD_loss == "prec_ori":
            return precision_ori_loss
# code for homogeneous masking
# input: y0-y1 w/ size of batch_size, time_step, hidden_size
def MaskOffNGroups(input, num_per_group, NUM_OFF):                  # force to turn off N groups
    batch_size, time_step, hidden_size = input.size()               # input: y0-y1 w/ size of [batch_size, time_step, hidden_size]
    #print(batch_size, time_step, hidden_size)

    batch_masks = torch.tensor([])                                  # masks of this batch
    for i in range(batch_size):                                     # for each sample
        time_step_masks = torch.tensor([])                          # masks of this sample
        for j in range(time_step):                                  # for each time-step
            s = input[i, j, :]                                      # i-th sample in this batch, j-th score in time series
            #print(s)
            sorted_s = sorted(range(len(s)), key = lambda k : s[k]) # value由小到大的index，小的應該要關
            mask = torch.ones(hidden_size)                          # force to turn on all nodes
            mask[sorted_s[:int(NUM_OFF * num_per_group)]] = 0       # 由小排到大的前 NUM_OFF 組(每組有 num_per_group 個node)關掉
            mask = mask[None, :]                                    # add 1 dim to append
            #print(mask)
            time_step_masks = torch.cat((time_step_masks, mask), 0) # append
        # time_step_masks.size() = time_step, hidden_size
        #print("time_step_masks.size(): ", time_step_masks.size())
        time_step_masks = time_step_masks[None, :]                  # add 1 dim to append
        batch_masks = torch.cat((batch_masks, time_step_masks), 0)  # append
    #batch_masks.size() = batch_size, time_step, hidden_size
    #print("batch_masks.size(): ", batch_masks.size())
    return batch_masks

# code for aggressive & passive masking
# input: y0-y1 w/ size of batch_size, time_step, hidden_size
# mask_ori: original mask formed by score passing Gumbel_softmax
# ratio: how much more should be toggle on/off, 0-1
# AGG: 1 for aggressive; 0 for passive
def AGG_PAS_masking(input, mask_ori, ratio, AGG):
  batch_size, time_step, hidden_size = input.size()                 # input: y0-y1 w/ size of batch_size, time_step, hidden_size
  #print(batch_size, time_step, hidden_size)

  batch_size, time_step, hidden_size = mask_ori.size()              # original mask formed by score passing Gumbel_softmax
  #print(batch_size, time_step, hidden_size)

  batch_masks = torch.tensor([])                                    # masks of this batch
  for i in range(batch_size):                                       # for each sample
    time_step_masks = torch.tensor([])                              # masks of this sample
    for j in range(time_step):                                      # for each time-step
      s = input[i, j, :]                                            # i-th sample in this batch, j-th score in time series
      m = mask_ori[i, j, :]                                         # i-th sample in this batch, j-th mask in time series
      #print("mask_ori: ", m)
      if AGG:                                                       # aggressive masking
        sorted_s = sorted(range(len(s)), key = lambda k : s[k])     # value由小到大的index，小的應該要關
        # 原本關的關；原本開的 挑出其中一定比例(ratio)關掉、照排序去關
        mask = torch.ones(hidden_size)                              # 先都開
        if ratio == 1:                                              # 全關
          mask = torch.zeros(hidden_size)
        else:
          N = int(m.sum() * ratio)                                  # num of nodes forced to be closed
          k = 0                                                     # num of nodes that has been changed from "open" to "close"
          for idx in range(len(s)):                                 # go through all nodes
            # 由小到大的index，小的應該要關
            if m[sorted_s[idx]] == 0:                               # 原本關的
              mask[sorted_s[idx]] = 0                               # 關
            else:                                                   # 原本開的，小的優先關
              if k < N:                                             # toggle off N more
                mask[sorted_s[idx]] = 0
                k += 1                                              # 多關了一個
      
      else:                                                         # passive masking
        sorted_s = sorted(range(len(s)), key = lambda k : s[k], reverse=True) 
                                                                    # value由大到小的index，大的應該要開
        # 原本開的開，原本關的 挑出一定比例(ratio)開起來、按順序
        mask = torch.zeros(hidden_size)                             # 先都關
        if ratio == 1:                                              # 全開
          mask = torch.ones(hidden_size)
        else:
          N = int((hidden_size - m.sum()) * ratio)                  # num of nodes forced to be opened
          #print("N = ", N)
          k = 0                                                     # num of nodes that has been changed from "close" to "open"
          for idx in range(len(s)):                                 # go through all nodes
            # 由大到小的index，大的應該要開
            if m[sorted_s[idx]] == 1:                               # 原本開的
              mask[sorted_s[idx]] = 1                               # 開
            else:                                                   # 原本關的，大的優先開
              if k < N:                                             # toggle on N more
                mask[sorted_s[idx]] = 1
                k += 1                                              # 多開了一個
      mask = mask[None, :]                                          # add 1 dim to append
      #print(mask)
      time_step_masks = torch.cat((time_step_masks, mask), 0)       # append
    # time_step_masks.size() = time_step, hidden_size
    #print(time_step_masks.size())
    time_step_masks = time_step_masks[None, :]                      # add 1 dim to append
    batch_masks = torch.cat((batch_masks, time_step_masks), 0)      # append
    #batch_masks.size() = batch_size, time_step, hidden_size
  return batch_masks

class Data2VecAudioForCTC(Data2VecAudioPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.data2vec_audio = Data2VecAudioModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Data2VecAudioForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.alpha=torch.tensor(LAMBDA)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("lm_thres = ", self.lm_thres)

        # 加toggle network, lm_model
        #self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.arbitrator = nn.Linear(config.hidden_size, config.hidden_size*4)    # 2條保護AD資訊（one-hot後用其中一條），2條保護ASR資訊（one-hot後用其中一條）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        
        # 加dementia model
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        
        # define similarity loss: AM-Softmax, aka div loss
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        if STAGE == 1:                                                  # freeze all, train AD classifier alone
            print("Current stage: 1")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            #self.freeze_lm_fsm()
            self.freeze_arbitrator()
            self.freeze_criterion_similar()
        elif STAGE == 2:                                                # freeze all, train toggle network alone
            print("Current stage: 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()
        elif STAGE == 3:                                                # freeze all, train FSM + classifiers
            print("Current stage: 3")
            self.freeze_data2vec_audio()
            self.freeze_criterion_similar()           

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec_audio.feature_extractor._freeze_parameters()
    
    def freeze_data2vec_audio(self):
        self.data2vec_audio.eval()
        for param in self.data2vec_audio.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
    """        
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
    
    def freeze_arbitrator(self):
        self.arbitrator.eval()
        for param in self.arbitrator.parameters():
            param.requires_grad = False       


    @add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.data2vec_audio(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 沒過FSM，用來單獨train AD classifier
        dementia_logits_unmask = self.dementia_head(hidden_states) # for stage 1 training
        
        # hidden_states: data2vec_audio embedding
        ###################
        # 製造mask
        ###################
        """
        m = nn.Sigmoid()
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask) # to has grad?
        """
        #m = nn.Sigmoid()
        #all_score = m(self.arbitrator(hidden_states))             # score range from 0~1
        all_score = self.arbitrator(hidden_states)
        """
        all_mask = torch.where(all_score >= self.lm_thres.to(all_score.device), torch.tensor(1.0).to(all_score.device), torch.tensor(0.0).to(all_score.device))                   # if condition, 1. else, 0
        all_mask = all_mask + 0 * self.arbitrator(hidden_states) # to have grad?  
        """
        # use Gunbel softmax
        #print(all_score)
        lm_score = torch.stack((all_score[:, :, :self.config.hidden_size] , all_score[:, :, self.config.hidden_size:self.config.hidden_size*2]), -1)     # first part for lm, size = [batch_size, time-step, hidden_state, 2]
        AD_score = torch.stack((all_score[:, :, self.config.hidden_size*2:self.config.hidden_size*3] , all_score[:, :, self.config.hidden_size*3:]), -1) # second part for AD, size = [batch_size, time-step, hidden_state, 2]

        # toggle ratio
        if TOGGLE_RATIO != 0:                                                           # if toggle ratio is set
            # lm_score
            y0 = lm_score[:, :, :, 0]                                                   # target vector
            y1 = lm_score[:, :, :, 1]                                                   # another vector
            lm_score[:, :, :, 0] = (y1 - y0) * TOGGLE_RATIO + y0                        # replace target vector
            # AD_score
            y0 = AD_score[:, :, :, 0]                                                   # target vector
            y1 = AD_score[:, :, :, 1]                                                   # another vector
            AD_score[:, :, :, 0] = (y1 - y0) * TOGGLE_RATIO + y0                        # replace target vector      
        
        # mask by score passing GS
        #lm_mask = torch.nn.functional.gumbel_softmax(lm_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        lm_mask = gumbel_softmax(lm_score, tau=GS_TAU, hard=True, dim=-1)[:, :, :, 0]
        #AD_mask = torch.nn.functional.gumbel_softmax(AD_score, hard=True, dim=-1)[:, :, :, 0] # back to [batch_size, time-step, hidden_state]
        AD_mask = gumbel_softmax(AD_score, tau=GS_TAU, hard=True, dim=-1)[:, :, :, 0]
        
        if EXP_TYPE == 'h':                                                             # homogeneous masking
            num_per_group = int(self.config.hidden_size / 16)                           # divided into 16 groups
            delta_y = lm_score[:, :, :, 0] - lm_score[:, :, :, 1]                       # y0-y1 for lm
            lm_mask = MaskOffNGroups(delta_y, num_per_group, NUM_OFF)
            delta_y = AD_score[:, :, :, 0] - AD_score[:, :, :, 1]                       # y0-y1 for AD
            AD_mask = MaskOffNGroups(delta_y, num_per_group, NUM_OFF)           
        elif EXP_TYPE == 'a':                                                           # aggressive masking
            delta_y = lm_score[:, :, :, 0] - lm_score[:, :, :, 1]                       # y0-y1 for lm
            lm_mask = AGG_PAS_masking(delta_y, lm_mask, ratio=AP_RATIO, AGG=1)
            #print("lm_mask size: ", lm_mask.size())
            delta_y = AD_score[:, :, :, 0] - AD_score[:, :, :, 1]                       # y0-y1 for AD
            AD_mask = AGG_PAS_masking(delta_y, AD_mask, ratio=AP_RATIO, AGG=1)
            #print("AD_mask size: ", AD_mask.size())
            #aaa=ccc
        elif EXP_TYPE == 'p':                                                           # passive masking
            delta_y = lm_score[:, :, :, 0] - lm_score[:, :, :, 1]                       # y0-y1 for lm
            lm_mask = AGG_PAS_masking(delta_y, lm_mask, ratio=AP_RATIO, AGG=0)
            #print("lm_mask size: ", lm_mask.size())
            delta_y = AD_score[:, :, :, 0] - AD_score[:, :, :, 1]                       # y0-y1 for AD
            AD_mask = AGG_PAS_masking(delta_y, AD_mask, ratio=AP_RATIO, AGG=0)
            #print("AD_mask size: ", AD_mask.size())
            #aaa=ccc
        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        """
        lm_masked = lm_mask*hidden_states
        """
        lm_masked = lm_mask*hidden_states
        AD_masked = AD_mask*hidden_states

        ##############
        # head(clf)
        ##############
        """
        logits = self.lm_head(lm_masked)
        dementia_logits = self.dementia_head(lm_masked) # masked hidden state 過AD classifier
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)
        dementia_output_mean = torch.mean(dementia_logits_unmask,dim=1)
        """
        logits = self.lm_head(lm_masked)                                                    # ASR loss
        dementia_logits = self.dementia_head(lm_masked)                                     # for AD GRL
        
        dementia_output_mean_2r = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = ReverseLayerF.apply(dementia_output_mean_2r, self.alpha)   # for AD GRL
        dementia_output_mean_unmask = torch.mean(dementia_logits_unmask,dim=1)              # unmask

        logits_r = self.lm_head(AD_masked)                                                  # for ASR GRL
        dementia_logits = self.dementia_head(AD_masked)                                     # for AD classifier
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1) # logit轉prob
            log_probs_r = ReverseLayerF.apply(log_probs_r, self.alpha) # ASR-GRL
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # ASR GRL
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
                if AD_loss == "cel":
                    print("loss: cel")
                    loss_fn = nn.CrossEntropyLoss()
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)                # reverse
                elif AD_loss == "recall":                 
                    #print("loss: recall")
                    loss_fn = RecallLoss(weight=W_LOSS)                                                 # W_LOSS=[w_HC, w_AD]
                    #loss = criterion(y_predict, y_target)
                    # predict: [N, C, *]    ; target: [N, *]
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier: [batch_size, 2], [batch_size,]
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask: [batch_size, 2], [batch_size,]
                    #print("dementia_output_mean_unmask: ", dementia_output_mean_unmask)
                    #print("dementia_labels: ", dementia_labels)
                    #print("dementia_loss: ", dementia_loss_unmask)
                    
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse: [batch_size, 2], [batch_size,]

                elif AD_loss == "prec":                 
                    #print("loss: precision")
                    loss_fn = RecallLoss(weight=[0.1, 0.9])                                                      # emphasize on AD PAR
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse
                elif AD_loss == "f1":
                    #print("loss: f1")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse         
                elif AD_loss == "prec_ori":     
                    #print("loss: prec_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse     
                elif AD_loss == "recall_ori":     
                    #print("loss: recall_ori")
                    loss_fn = RecallLoss(weight=[0.5, 0.5])
                    
                    dementia_loss = loss_fn(dementia_output_mean, dementia_labels, AD_loss)                      # AD classifier
                    dementia_loss_unmask = loss_fn(dementia_output_mean_unmask, dementia_labels, AD_loss)        # unmask
                    dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels, AD_loss)                # reverse     
                # att loss
                Att_loss = FSMatt_loss(lm_mask, AD_mask)                                                        # not used in this version
                # diversity loss: AM-Softmax
                #scores = torch.cat((hidden_states * lm_mask, hidden_states * AD_mask), dim=0)
                #am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * AD_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # batch_size*time-step, hidden_size
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # batch_size*time-step, hidden_size
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # batch_size*time-step * 2, hidden_size
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # batch_size*time-step * 2
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)
                #print("========================")
                #print(AD_mask, lm_mask)
                #print(loss, dementia_loss_rev, loss_r, dementia_loss, Att_loss, score_loss)

                if STAGE == 1:                                                  # train AD classifier
                    #print("Current stage: 1")
                    final_loss = dementia_loss_unmask
                elif STAGE == 2:                                                # train toggle network
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + score_loss #+ Att_loss #+ score_loss
                    #print(loss, dementia_loss_rev, loss_r, dementia_loss, l2_lambda * l2_norm)
                    #final_loss = loss + dementia_loss_rev + loss_r + dementia_loss + l2_lambda * l2_norm
                    #final_loss = l2_lambda * l2_norm
                elif STAGE == 3:
                    final_loss = loss + dementia_loss_rev + loss_r + dementia_loss #+ Att_loss + score_loss
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        # return info that we might need
        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, "dementia_mask": AD_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.hubert = HubertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()
        
        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_hubert()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_hubert()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_hubert()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
    
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()
    
    def freeze_hubert(self):
        self.hubert.eval()
        for param in self.hubert.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
            
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
            
    def freeze_dementia_fsm(self):
        self.dementia_fsm.eval()
        for param in self.dementia_fsm.parameters():
            param.requires_grad = False
            
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
   
    def freeze_lm_grl(self):
        self.lm_grl.eval()
        for param in self.lm_grl.parameters():
            param.requires_grad = False
 
    def freeze_dementia_grl(self):
        self.dementia_grl.eval()
        for param in self.dementia_grl.parameters():
            param.requires_grad = False
 

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: hubert embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        #del dementia_logits_r, dementia_logits
        del dementia_logits_r
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # loss for lm_grl
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if STAGE == 1:                                                  # train FSM
                    #print("Current stage: 1")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 2:                                                # train ASR
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 3:                                                # train dementia GRL
                    #print("Current stage: 3")
                    final_loss = dementia_loss_rev
                elif STAGE == 4:
                    final_loss = loss_r
                elif STAGE == 5:
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class SEWDForCTC(SEWDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.sew_d = SEWDModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWDForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)

        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()

        
        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_sew_d()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_sew_d()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_sew_d()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()

            
        # Initialize weights and apply final processing
        self.post_init()
    
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_sew_d(self):
        self.sew_d.eval()
        for param in self.sew_d.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
            
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
            
    def freeze_dementia_fsm(self):
        self.dementia_fsm.eval()
        for param in self.dementia_fsm.parameters():
            param.requires_grad = False
            
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
   
    def freeze_lm_grl(self):
        self.lm_grl.eval()
        for param in self.lm_grl.parameters():
            param.requires_grad = False
 
    def freeze_dementia_grl(self):
        self.dementia_grl.eval()
        for param in self.dementia_grl.parameters():
            param.requires_grad = False
  

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.sew_d(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: sew_d embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        #del dementia_logits_r, dementia_logits
        del dementia_logits_r
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # loss for lm_grl
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if STAGE == 1:                                                  # train FSM
                    #print("Current stage: 1")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 2:                                                # train ASR
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 3:                                                # train dementia GRL
                    #print("Current stage: 3")
                    final_loss = dementia_loss_rev
                elif STAGE == 4:
                    final_loss = loss_r
                elif STAGE == 5:
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                    # train dementia_FSM
                # ////

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class UniSpeechSatForCTC(UniSpeechSatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechSatForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        self.alpha=torch.tensor(LAMBDA)
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)
        
        # 加lm_model
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)                             # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia model
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor
        self.freeze_feature_encoder()

        if STAGE == 1:                                                  # train FSM
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                # train dementia GRL
            print("Current stage: 3")
            self.freeze_unispeech_sat()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                # train lm GRL
            print("Current stage: 4")
            self.freeze_unispeech_sat()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                # train encoder
            # train lm_FSM
            self.freeze_unispeech_sat()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
         
        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    def freeze_unispeech_sat(self):
        self.unispeech_sat.eval()
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False
    
    def freeze_criterion_similar(self):
        self.criterion_similar.eval()
        for param in self.criterion_similar.parameters():
            param.requires_grad = False
            
    def freeze_lm_fsm(self):
        self.lm_fsm.eval()
        for param in self.lm_fsm.parameters():
            param.requires_grad = False
            
    def freeze_dementia_fsm(self):
        self.dementia_fsm.eval()
        for param in self.dementia_fsm.parameters():
            param.requires_grad = False
            
    def freeze_lm_head(self):
        self.lm_head.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_dementia_head(self):
        self.dementia_head.eval()
        for param in self.dementia_head.parameters():
            param.requires_grad = False
   
    def freeze_lm_grl(self):
        self.lm_grl.eval()
        for param in self.lm_grl.parameters():
            param.requires_grad = False
 
    def freeze_dementia_grl(self):
        self.dementia_grl.eval()
        for param in self.dementia_grl.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_PROCESSOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        dementia_labels=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]                   # last_hidden_state
        hidden_states = self.dropout(hidden_states)

        # hidden_states: unispeech_sat embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states)) # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))             # score range from 0~1
        
        #model.recognize(inputs, input_lengths)
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device)) # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                   # if condition, 1. else, 0
        
        # 拿score vector 跟原本的hidden_states點乘
        dementia_resored = dementia_score*hidden_states
        lm_resored = lm_score*hidden_states
        
        # head(clf)
        dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_resored)
        # del dementia_resored, lm_resored
        
        # grl(dis)
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        dementia_score_r = m(self.dementia_fsm(hidden_states_r)) # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))            # score range from 0~1

        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        # 拿mask跟hidden_states_r點乘
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        dementia_output_mean = torch.mean(dementia_logits,dim=1)
        dementia_output_mean_r = torch.mean(dementia_logits_r,dim=1)
        #del dementia_logits_r, dementia_logits
        del dementia_logits_r
        #*******************
        
        final_loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            log_probs_r = nn.functional.log_softmax(logits_r, dim=-1, dtype=torch.float32).transpose(0, 1)
 
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # loss for lm_grl
                loss_r = nn.functional.ctc_loss(
                    log_probs_r,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                #  /////
                # gradient reversal layers(GRL)
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # multi-task
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                scores = torch.cat((hidden_states * lm_score, hidden_states * dementia_score), dim=0)
                del lm_score, dementia_score
                am_labels = torch.cat((torch.zeros(len(hidden_states), dtype=torch.long), torch.ones(len(hidden_states), dtype=torch.long)), dim=0).to('cpu')
                #del hidden_states
                similarity, _ = self.criterion_similar(scores, am_labels)
                score_loss = similarity # * args.w_score if args.w_score > 0. else torch.tensor(0.).to(device)

                if STAGE == 1:                                                  # train FSM
                    #print("Current stage: 1")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 2:                                                # train ASR
                    #print("Current stage: 2")
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                elif STAGE == 3:                                                # train dementia GRL
                    #print("Current stage: 3")
                    final_loss = dementia_loss_rev
                elif STAGE == 4:
                    final_loss = loss_r
                elif STAGE == 5:
                    # train lm_FSM
                    final_loss = loss + dementia_loss_rev
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        logits_all = {'ASR logits': logits, 'dementia logits': dementia_logits, 'hidden_states': hidden_states,
                    'lm_mask': lm_mask, 'dementia_mask': dementia_mask}

        return CausalLMOutput(
            loss=final_loss, logits=logits_all, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

def map_to_result(batch, idx):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"]).unsqueeze(0)            
        logits = model(input_values).logits                                     # includes ASR logits, dementia logits, hidden_states
        asr_lg = logits['ASR logits']
        AD_lg = logits['dementia logits'][0]                                    # (1, time-step, 2) --> (time-step, 2)
    
    pred_ad_tstep = torch.argmax(AD_lg, dim=-1)                                 # pred of each time-step
    pred_ad = pred_ad_tstep.sum() / pred_ad_tstep.size()[0]                     # average result
    if pred_ad > 0.5:                                                           # if over half of the time pred AD
        batch["pred_AD"] = 1                                                    # save final result as AD
    else:
        batch["pred_AD"] = 0
    
    pred_ids = torch.argmax(asr_lg, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]                     # predicted transcript
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)       # ground truth transcript
    
    # for toggle
    df = pd.DataFrame({'path': batch["path"],                                   # to know which sample
                       'array': str(batch["array"]),
                       'text': batch["text"],                                   # ground truth transcript
                       'dementia_labels': batch["dementia_labels"],
                       'input_values': str(batch["input_values"]),              # input of the model
                       'labels': str(batch["labels"]),
                       'ASR logits': str(logits["ASR logits"].tolist()),
                       'dementia logits': str(logits["dementia logits"].tolist()),
                       'hidden_states': str(logits["hidden_states"].tolist()),
                       'pred_AD': batch["pred_AD"],                             # AD prediction
                       'pred_str': batch["pred_str"],                           # predicted transcript
                       'dementia_mask': str(logits["dementia_mask"].tolist()),  # ASR-free mask for AD classification
                       'lm_mask': str(logits["lm_mask"].tolist())},             # AD-free mask for ASR task
                      index=[idx])
    
    """
    # for model w.o. FSM
    df = pd.DataFrame({'path': batch["path"],                                    # to know which sample
                    'array': str(batch["array"]),
                    'text': batch["text"],
                    'dementia_labels': batch["dementia_labels"],
                    'input_values': str(batch["input_values"]),               # input of the model
                    'labels': str(batch["labels"]),
                    'ASR logits': str(logits["ASR logits"].tolist()),
                    'dementia logits': str(logits["dementia logits"].tolist()),
                    'hidden_states': str(logits["hidden_states"].tolist()),
                    'pred_AD': batch["pred_AD"],                             # AD prediction
                    'pred_str': batch["pred_str"]},
                    index=[idx])
    """
    return df

from transformers import Data2VecAudioConfig, HubertConfig, SEWDConfig, UniSpeechSatConfig

# load according to model type
# note that only data2vec is done for this version
if model_type == "wav2vec":
    model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    name = "facebook/wav2vec2-base-960h" # + model_dir.split("/")[-3]
    print("Current model: ", name)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "data2vec":
    name = "facebook/data2vec-audio-large-960h" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config to avoid code from stopping
    config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = Data2VecAudioForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "hubert":
    name = "facebook/hubert-xlarge-ls960-ft" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = HubertConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = HubertForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "sewd":
    name = "asapp/sew-d-mid-400k-ft-ls100h" #+ model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = SEWDConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = SEWDForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)
elif model_type == "unispeech":
    name = "microsoft/unispeech-sat-base-100h-libri-ft" # + model_in_dir.split("/")[-3]
    print("Current model: ", name)
    mask_time_prob = 0                                                                     # change config
    config = UniSpeechSatConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
    model = UniSpeechSatForCTC.from_pretrained(model_dir, config=config)
    processor = Wav2Vec2Processor.from_pretrained(name)   


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# store result of test data
test_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")
test_data = test_data.map(prepare_dataset, num_proc=10)

# get emb.s, masks... 1 sample by 1 sample
df = map_to_result(test_data[0], 0)
for i in range(len(test_data) - 1):
    df2 = map_to_result(test_data[i+1], i+1)
    df = pd.concat([df, df2], ignore_index=True)
    print("\r"+ str(i), end="")

csv_path = "./saves/results/" + csv_name + ".csv"
df.to_csv(csv_path)
print("Testing data Done")


# store result of train data
train_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/train.csv")
train_data = train_data.map(prepare_dataset, num_proc=10)

# get emb.s, masks... 1 sample by 1 sample
df = map_to_result(train_data[0], 0)
for i in range(len(train_data) - 1):
    df2 = map_to_result(train_data[i+1], i+1)
    df = pd.concat([df, df2], ignore_index=True)
    print("\r"+ str(i), end="")

csv_path = "./saves/results/" + csv_name + "_train.csv"
df.to_csv(csv_path)
print("Training data Done")

# store result of dev data
"""
dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
dev_data = dev_data.map(prepare_dataset, num_proc=10)

df = map_to_result(dev_data[0], 0)
for i in range(len(dev_data) - 1):
    df2 = map_to_result(dev_data[i+1], i+1)
    df = pd.concat([df, df2], ignore_index=True)
    print("\r"+ str(i), end="")

csv_path = "./saves/results/" + csv_name + "_dev.csv"
df.to_csv(csv_path)
"""
print(csv_name + " All Done")
