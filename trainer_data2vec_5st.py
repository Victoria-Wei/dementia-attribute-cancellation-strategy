import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers import Wav2Vec2Processor, Data2VecAudioModel
from transformers.models.data2vec.modeling_data2vec_audio import Data2VecAudioFeatureProjection, Data2VecAudioPositionalConvLayer, Data2VecAudioEncoder, Data2VecAudioFeatureEncoder

import math
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.training_args import TrainingArguments
from transformers.utils import logging

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import torch_int_div
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.data2vec.configuration_data2vec_audio import Data2VecAudioConfig

_PROCESSOR_FOR_DOC = "Wav2Vec2Processor"
_CHECKPOINT_FOR_DOC = "facebook/data2vec-audio-base-960h"

_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 66.95

from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
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

class Data2VecAudioPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecAudioConfig
    base_model_prefix = "data2vec_audio"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Data2VecAudioFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, Data2VecAudioPositionalConvLayer):
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feat_extract_output_lengths with
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch_int_div(input_length - kernel_size, stride) + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PreTrainedModel._get_feature_vector_attention_mask
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Data2VecAudioEncoder, Data2VecAudioFeatureEncoder)):
            module.gradient_checkpointing = value


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


_CONFIG_FOR_DOC = "Data2VecAudioConfig"
logger = logging.get_logger(__name__)

    
import pandas as pd
from datasets import Dataset
import librosa
from datasets import load_from_disk
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

from transformers import Trainer
from transformers import TrainingArguments
import json

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
        self.dementia_thres = torch.tensor(AD_THRES)
        self.lm_thres = torch.tensor(LM_THRES)
        print("lambda = ", self.alpha)
        print("dementia_thres = ", self.dementia_thres)
        print("lm_thres = ", self.lm_thres)

        # 加lm相關components
        self.lm_fsm = nn.Linear(config.hidden_size, config.hidden_size)          # 找出對lm重要的feat
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)          # output字母的"機率"
        self.lm_grl = nn.Linear(config.hidden_size, config.vocab_size)           # 加了GRL那條
        
        # 加dementia相關components
        self.dementia_fsm = nn.Linear(config.hidden_size, config.hidden_size)    # 找出對AD預測重要的feat
        self.dementia_head = nn.Linear(config.hidden_size, 2)                    # 辨識AD
        self.dementia_grl = nn.Linear(config.hidden_size, 2)                     # 加GRL那條
        
        # define similarity loss: AM-Softmax, aka div loss
        self.criterion_similar = AngularPenaltySMLoss(in_features=config.hidden_size, out_features=2, loss_type='cosface').to('cpu')
        
        # freeze feature_extractor    
        self.freeze_feature_encoder()

        # skip to stage 6
        if STAGE == 1:                                                           # train FSMs alone
            print("Current stage: 1")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            self.freeze_lm_head()
            self.freeze_dementia_head()
        elif STAGE == 2:                                                         # train FSM + head
            print("Current stage: 2")
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 3:                                                         # train dementia GRL
            print("Current stage: 3")
            self.freeze_data2vec_audio()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_lm_grl()
        elif STAGE == 4:                                                         # train lm GRL
            print("Current stage: 4")
            self.freeze_data2vec_audio()
            self.freeze_lm_fsm()
            self.freeze_dementia_fsm()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_dementia_grl()
        elif STAGE == 5:                                                         # train lm_FSM
            self.freeze_data2vec_audio()
            self.freeze_dementia_fsm()            
            self.freeze_criterion_similar()
            self.freeze_lm_head()
            self.freeze_dementia_head()            
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
        elif STAGE == 6:                                                         # train 2 FSM
            print("Current stage: new 2")
            self.freeze_data2vec_audio()
            self.freeze_lm_head()
            self.freeze_dementia_head()
            self.freeze_criterion_similar()
            self.freeze_lm_grl()
            self.freeze_dementia_grl()
            
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

        # hidden_states: data2vec_audio embedding
        # 製造mask
        m = nn.Sigmoid()
        dementia_score = m(self.dementia_fsm(hidden_states))            # score range from 0~1
        lm_score = m(self.lm_fsm(hidden_states))                        # score range from 0~1
        
        # if score >= thredhold, mask = 1
        dementia_mask = torch.where(dementia_score >= self.dementia_thres.to(dementia_score.device), torch.tensor(1.0).to(dementia_score.device), torch.tensor(0.0).to(dementia_score.device))  # if condition, 1. else, 0
        lm_mask = torch.where(lm_score >= self.lm_thres.to(lm_score.device), torch.tensor(1.0).to(lm_score.device), torch.tensor(0.0).to(lm_score.device))                                      # if condition, 1. else, 0
        lm_mask = lm_mask + 0 * self.lm_fsm(lm_mask)                    # to has grad?
        dementia_mask = dementia_mask + 0 * self.lm_fsm(lm_mask)        # to has grad?

        # 拿score vector 跟原本的hidden_states點乘
        #dementia_resored = dementia_score*hidden_states
        #lm_resored = lm_score*hidden_states

        ##################################
        # 拿mask跟原本的hidden_states點乘 #
        ##################################
        dementia_masked = dementia_mask*hidden_states
        lm_masked = lm_mask*hidden_states
        
        ##############
        # head(clf)
        ##############
        #dementia_logits = self.dementia_head(dementia_resored) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        #logits = self.lm_head(lm_resored)
        dementia_logits = self.dementia_head(dementia_masked) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits = self.lm_head(lm_masked)
        # del dementia_resored, lm_resored
        dementia_output_mean = torch.mean(dementia_logits,dim=1)

        ##############
        # grl(dis)
        ##############
        hidden_states_r = ReverseLayerF.apply(hidden_states, self.alpha)
        # get score from reversed embedding
        dementia_score_r = m(self.dementia_fsm(hidden_states_r))            # score range from 0~1
        lm_score_r = m(self.lm_fsm(hidden_states_r))                        # score range from 0~1
        # if score >= thredhold, mask = 1
        dementia_mask_r = torch.where(dementia_score_r >= self.dementia_thres.to(dementia_score_r.device), torch.tensor(1.0).to(dementia_score_r.device), torch.tensor(0.0).to(dementia_score_r.device)) # if condition, 1. else, 0
        lm_mask_r = torch.where(lm_score_r >= self.lm_thres.to(lm_score_r.device), torch.tensor(1.0).to(lm_score_r.device), torch.tensor(0.0).to(lm_score_r.device))                   # if condition, 1. else, 0
        
        del dementia_score_r, lm_score_r
        #####################################
        # 拿mask跟reversed hidden_states點乘 #
        #####################################
        dementia_masked_r = dementia_mask_r*hidden_states_r
        lm_masked_r = lm_mask_r*hidden_states_r
        
        del hidden_states_r, dementia_mask_r, lm_mask_r
        # grl(dis)
        dementia_logits_r = self.dementia_grl(lm_masked_r) #******************* torch.Size([2, 1327, 32]) (batchsize, timestep, feature_dimention)
        logits_r = self.lm_grl(dementia_masked_r)
        del dementia_masked_r, lm_masked_r
        
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
                
                loss_fn = nn.CrossEntropyLoss()
                
                dementia_loss = loss_fn(dementia_output_mean, dementia_labels)        # loss for AD
                dementia_loss_rev = loss_fn(dementia_output_mean_r, dementia_labels)  # AD-GRL
                
                # FSM att loss
                # Scorematrix = append([dementia_mask,lm_mask]) # torch.Size([2, embedding_size])
                # Att_loss = Scorematrix*Scorematrix - Identity matrix
                #Att_loss = FSMatt_loss(lm_score, dementia_score)
                Att_loss = FSMatt_loss(lm_mask, dementia_mask)                        # use mask to compute attention loss
                # del lm_mask, dementia_mask
                # diversity loss: AM-Softmax
                lm_masked = hidden_states * lm_mask
                AD_masked = hidden_states * dementia_mask
                lm_masked = torch.reshape(lm_masked, (lm_masked.size()[0]*lm_masked.size()[1], lm_masked.size()[2])) # to size [batch_size*time-step, hidden_size]
                AD_masked = torch.reshape(AD_masked, (AD_masked.size()[0]*AD_masked.size()[1], AD_masked.size()[2])) # to size [batch_size*time-step, hidden_size]
                #print("lm_masked size: ", lm_masked.size())
                #print("AD_masked size: ", AD_masked.size())

                scores = torch.cat((lm_masked, AD_masked), dim=0) # size: [batch_size*time-step * 2, hidden_size]
                #print("score size: ", scores.size())
                am_labels = torch.cat((torch.zeros(len(lm_masked), dtype=torch.long), torch.ones(len(AD_masked), dtype=torch.long)), dim=0).to('cpu') # [batch_size*time-step * 2] w/ 1st half being 0s, and 2nd half being 1s
                #print("am_labels size: ", am_labels.size())
                #print(am_labels)

                # should feed x: [batch_size, hidden_size] & labels: [batch_size] simply use num, no need to one-hot
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
                elif STAGE == 6:                                                # ASR loss, AD Loss (CE), diversity loss, and attention loss
                    final_loss = loss + dementia_loss + score_loss + Att_loss
                # ////
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]

        return CausalLMOutput(
            loss=final_loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    
class CustomTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """
            #dementia_labels = inputs.pop("dementia_labels") # pop 出來就會不見?
            
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            
            outputs = model(**inputs)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        
        # write to txt file
        file_object = open(LOG_DIR + log_file, 'a')
        # Append at the end of file
        file_object.write(json.dumps(output) + '\n')
        # Close the file
        file_object.close()

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


        
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lam', '--LAMBDA', type=float, default=0.5, help="Lambda for GRL")
parser.add_argument('-st', '--STAGE', type=int, default=1, help="Current training stage")
parser.add_argument('-GRL', '--GRL', action='store_true', default=False, help="True: GRL")
parser.add_argument('-model_in', '--model_in_path', type=str, default="./saves/wav2vec2-base-960h_GRL_0.5/checkpoint-14010/", help="Where the model is saved")
parser.add_argument('-model_out', '--model_out_path', type=str, default="./saves/wav2vec2-base-960h_linear_GRL", help="Where to save the model")
parser.add_argument('-log', '--log_path', type=str, default="wav2vec2-base-960h_linear_GRL.txt", help="name for the txt file")
args = parser.parse_args()
LAMBDA = args.LAMBDA                    # lambda for GRL
REVERSE = args.GRL                      # not used in this version
STAGE = args.STAGE                      # stage 1: train AD classifier; stage 2: train toggling network
model_in_dir = args.model_in_path       # path to load the initial model
model_out_dir = args.model_out_path     # path to store the resulted model
log_file = args.log_path                # path to save log file

# 設定log file位置與名稱
LOG_DIR = './saves/log/'

# threshold for maskes
AD_THRES = 0.5
LM_THRES = 0.5

# load model from huggingface hub, here data2vec model
name = "facebook/data2vec-audio-large-960h"# + model_in_dir.split("/")[-3]
print("Current model: ", name)
from transformers import Data2VecAudioConfig
mask_time_prob = 0                                         # change config to avoid training stopping
config = Data2VecAudioConfig.from_pretrained(name, mask_time_prob=mask_time_prob)
model = Data2VecAudioForCTC.from_pretrained(model_in_dir, config=config)
model.config.ctc_zero_infinity = True                      # to avoid inf values

processor = Wav2Vec2Processor.from_pretrained(name)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# load train / test data
train_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/train.csv")
#dev_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/dev.csv")
test_data = csv2dataset(path = "/mnt/Internal/FedASR/Data/ADReSS-IS2020-data/mid_csv/test.csv")

# map to desired form
train_data = train_data.map(prepare_dataset, num_proc=10)
#dev_data = dev_data.map(prepare_dataset, num_proc=10)
test_data = test_data.map(prepare_dataset, num_proc=10)

# skip to stage 6
if STAGE == 1:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=6,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 2:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=16,                 # finetune & GRL
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
elif STAGE == 6:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=30,                 # FSM alone
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-3, # 原本用1e-5
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        #adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5
    )
else:
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        num_train_epochs=40,
        fp16=True,
        gradient_checkpointing=True, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
        log_level='debug',
        logging_strategy="steps",
        adafactor=True,            # default:false. Whether or not to use transformers.Adafactor optimizer instead of transformers.AdamW
        #fp16_full_eval=True,      # to save memory
        #max_grad_norm=0.5,
    ) 

trainer = CustomTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=processor.feature_extractor,
)

trainer.train()
# save resulted model as "final"
trainer.save_model(model_out_dir + "/final")
