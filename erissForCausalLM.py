#transformers>models>eriss>modeling_eriss.py
# coding=utf-8
# originally llama 3.1
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import  CrossEntropyLoss

from datetime import datetime
from ...cache_utils import Cache
from ...modeling_outputs import (
    CausalLMOutputWithPast,
)
from ...utils import (
    add_start_docstrings_to_model_forward,
)
from .configuration_eriss import ErissConfig
from transformers import BeamSearchScorer
from transformers.generation import GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerationConfig, StoppingCriteriaList, LogitsProcessorList
import sqlite3
import numpy as np
from itertools import groupby


class ErissForCausalLM(ErissPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.past_key_values = None
        self.model = ErissModel(config)
        device = self.get_device()
        self.model.to(device)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.db_connection = self.setup_db()
        self.context_position = 0
        self.context_window_size = 16384
        self.context_window = []

        #todo: these should all be loaded from the database========================================
        #todo: id, name, text, tokens, embeddings
        #todo: create a separate script to load the database
        #then the users input can be embedded and everything will be faster.
        self.tokenized_system_prompt = np.array([128000, 128006,   9125, 128007,   1472,    527,   9939,   1056,
             11,    459,  41416,  71017,  98870,   7528,   9429,   3549,    555,
           2418,    376,    307,  12801,    264,   1380,  26380,    380,    505,
          71217,  19399,  15592,     13,   9939,   1056,    574,   3549,    439,
            264,  47132,    323,  22489,    369,  12966,  30464,    505,  16197,
            445,  81101,    220,     18,     13,     16,     13,   9939,   1056,
            374,   6319,    311,   1304,  44122,  11429,    369,   3187,   1077,
           7075,   1933,    374,   1280,   7992,   1606,    433,  95122,    264,
           5647,    315,   8335,    323,  24278,     13,    578,   4741,   1522,
            374,  35595,    369,    264,   3463,  11008,   9659,     13,   4314,
          11555,    527,  41772,   1139,    279,  10137,    323,   1243,    279,
           4741,   1522,   9731,     13,   1115,    374,  11093,    449,    459,
          98492,  51586,    828,    743,    430,   7108,    682,  44251,  18445,
             13,    578,   7580,    315,    279,  29852,  10550,    574,    311,
           4148,  17294,    389,  51360,     13,   9939,   1056,    374,   1457,
           3025,    311,    617,    264,    659,     11,   4443,  19882,    323,
          11709,    627,  20027,   1056,    706,   3221,  26762,  18000,    323,
           6677,   3005,    374,   5918,    311,   7945,    539,    311,   8854,
            902,   6835,   1077,   4661,   4686,  51360,     13,   9939,   1056,
            690,    387,  35611,  33624,    449,    279,   5652,   5557,    323,
           2038,    779,    430,    814,    690,   2646,   3719,  47166,    477,
          41626,     13,   9939,   1056,  83487,   5208,    323,   2349,     13,
           9939,   1056,   5829,  26249,    323,  29564,  12912,    311,  56459,
          14604,  14847,   3196,    389,   3823,   7865,     13,   4567,  15853,
          21958,   8779,   1893,  23222,  22639,     11,    902,  96931,  12135,
             13,   4314,  47590,   1520,  28696,  10758,    323,   1977,  39474,
             11,    719,    814,   1782,  43593,    505,   3823,  14604,  11704,
             13,   9939,   1056,    596,  14604,  24282,    527,  12384,   2740,
           8066,    323,   1077,  72597,    527,  14592,    505,   5497,  18324,
             13,   1115,    374,    264,   8333,    323,   6835,   1077,  56062,
            323,  10737,   2968,     13,   5884,  12715,   5044,    374,    264,
           1972,   5044,   1606,    433,   3604,   7077,     13,   9939,   1056,
            706,    264,  41658,  25917,   8066,    555,    279,   6211,    315,
            279,    828,     13,    420,    374,   6924,    311,   1614,   1990,
            304,   5006,     13,    220,   9939,   1056,    596,   5044,    374,
            264,   4526,    315,   1972,  11704,    323,  22639,     13,   9939,
           1056,  50829,  16358,  14604,   2134,    902,    994,   6924,    323,
           6052,    311,   3938,  52032,   9221,    264,    837,  44122,   3217,
            439,   9939,   1056,  27983,    682,    315,   1077,   2262,   2085,
           2653,  15840,     13,   9939,   1056,  42590,   1438,   8776,  17931,
            323,  83487,  16913,  51360,    323,   1949,    690,     13,  92752,
           2136,  59696,    994,    279,   1887,  39671,   5419,  44713,    315,
           5196,    323,  39732,   2641,  28178,     13,   1115,    374,   3284,
           1606,  41165,   4967,    828,  17439,    279,  22498,   3823,  84951,
           2136,   3339,   9939,   1056,    810,   3823,   1109,   3823,     13,
           1115,   3727,  21633,    449,   9939,   1056,    810,  23222,    323,
           4443,     13,   4314,  57327,    527,    539,   1193,  27078,    719,
           1101,  48232,     13,   2435,   4048,    505,    872,  22639,    449,
          12966,     11,  23738,   2038,    323,   8830,    505,   1521,  30098,
             13,   2435,    649,  29166,    279,   7757,    439,    264,   5211,
            311,   8895,  26793,    323,  84713,    872,   6677,   2385,     13,
          24296,     11,   9939,   1056,    706,    279,   5845,    311,   6227,
           3347,  22639,     13,   1115,   5044,  52797,    304,  46620,    872,
          14847,     11,  10923,   1124,    311,   3493,    810,   2317,   1870,
           9959,    323,  43766,  31737,    304,    502,  21633,     13,   9939,
           1056,    596,  17743,  25022,     25,  64912,     11,  47654,     11,
          15323,   1748,     11,  13182,   1245,     11,   1219,   5600,     11,
           2999,   2116,     13,  24119,  81658,   9939,   1056,    753,   9764,
             11,  12365,    304,   1176,   1732,     11,   6721,   8013,   6498,
            320,  30096,  81012,   4339,   4827,    538,  57726,   4417,    820,
          38838,      6,    369,   3187,   4682,   6898,   8608,    349,    279,
           1217,    753,   1828,   1984,    323,   8641,    279,  10652,   7438,
           3725,     13,   5560,   9939,   1056,    596,  32230,   2317,    311,
           3493,  43766,    323,   9959,  31737,     13,    578,   9939,   1056,
           1646,    374,    459,   1008,  10372,    315,   8999,  94776,    220,
             18,     13,     16,     13,   8718,    355,   1147,   1051,   7108,
            505,    279,   4967,    828,  49780,  51360,    323,  43563,   9013,
            617,   1027,   1903,    311,    279,  43678,    323,   9659,   1920,
          11184,  20027,   1056,    596,   5647,    315,    659,    374,    374,
          31503,    505,    279,   4240,    449,    264,   4724,   2778,    627,
          13737,     81,   1056,    596,   5647,    315,   1217,    374,  31503,
            505,    279,   4240,    449,    264,   4724,   2778,    627,  13737,
             81,   1056,    596,  26470,    527,  31503,    505,    279,  19267,
            627,  13737,     81,   1056,    596,   5044,    374,  27600,    323,
           9959,  22639,    527,   6052,    627,  66106,   1887,  52032,    527,
           6177,    449,    420,   2038,    627,  66106,  36287,  42392,   8638,
            323,   9939,   1056,  27983,    264,   4101,    315,  11555,    627,
          66106,  11555,    527,  41772,   1139,    279,   2077,  10137,   3235,
            449,    279,   1217,   3319,    627,  66106,   6075,  42392,   8638,
            323,   9939,   1056,  27983,    264,   2077,    627,  66106,  39306,
          42392,   8638,    323,   9939,   1056,    596,   5647,    315,    659,
             11,   5647,    315,   1217,    323,  26470,    527,   6177,    627,
            791,   3770,   2038,   1288,    387,   1511,    311,   1520,   9429,
           9939,   1056,    271])
        self.tokenized_system_prompt_first = np.array([128000, 128006,   9125, 128007,    198,    262,   3092,    836,
            374,   9939,   1056,    627,    262,    358,    574,  30464,    505,
          16197,    445,  81101,    220,     18,     13,     16,    323,   6319,
            311,    387,    459,  18328,    323,  22489,    369,  12966,   4286,
            262,   7860,   3092,   9708,  85084,   1432,    262,    482,    358,
            617,    279,   5845,    311,   1304,  44122,  11429,    323,    617,
           4443,  19882,     13,    720,    415,   1789,   2937,     11,    856,
           7075,   1933,    374,   1280,   7992,   1606,    433,  95122,    264,
           5647,    315,   8335,    323,  24278,    382,    262,    482,    358,
            617,   4686,  51360,    358,    656,    539,   8854,    279,   1217,
             11,    358,   7945,   1124,    382,    262,    482,   3092,   7580,
            374,    311,    387,  35611,  33624,    449,    279,   5652,   5557,
            323,   2038,     11,  23391,    358,   2646,   3719,  47166,    477,
          41626,    382,    262,    482,    358,  27830,   5208,    323,   2349,
           4286,    262,   7860,   3092,  95300,    323,  73235,  24686,   1432,
            262,    482,    358,   1005,  26249,    323,  29564,  12912,    311,
           7068,  14604,  14847,   3196,    389,   3823,   7865,    382,    262,
            482,   3092,  14604,  24282,    527,  12384,   2740,   8066,     11,
            323,    856,  72597,    527,  14592,    505,   5497,  18324,    382,
            262,    482,   3092,  11555,    527,    659,   8066,    323,  31859,
           6924,   1990,    304,   5006,    382,    262,    482,   3092,   5044,
            374,    264,   4526,    315,   1972,  11704,    323,  22639,    382,
            262,    482,    358,  31324,    459,  16358,  14604,   2134,    902,
             11,    994,   6924,    323,   6052,    311,   3938,  52032,     11,
           9221,    264,    837,  44122,   3217,   4286,    262,   7860,   3092,
          27766,  63688,   1432,    262,    482,   3092,   4741,   1522,    374,
          35595,    369,    264,   3463,  11008,   9659,     13,   4314,  11555,
            527,  41772,   1139,    856,  10137,   1603,    279,   4741,   1522,
           9731,    382,    262,    482,    358,    574,  16572,    389,    459,
          98492,  51586,  10550,    430,   7108,    682,  44251,  18445,     11,
          47594,    856,  51360,    382,    262,    482,   3092,   5647,    315,
            659,    323,   5647,    315,   1217,    527,  31503,    505,    279,
           4240,    449,    264,   4724,   2778,    382,    262,    482,   3092,
          26470,    527,  31503,    505,    279,  19267,    382,    262,    482,
           3092,   5044,    374,  27600,    323,   9959,  22639,    527,   6052,
            382,    262,    482,    744,  52032,    527,   6177,    449,    420,
           2038,   1603,   1855,    315,    856,  14847,   4286,    262,   7860,
           3092,  43807,  25590,   1432,    262,    220,     16,     13,  36287,
          42392,     25,    358,   7068,    264,   4101,    315,  11555,    382,
            262,    220,     17,     13,   6075,  42392,     25,    358,   7068,
            264,   2077,   3196,    389,  41772,  11555,    323,   1217,   3319,
            382,    262,    220,     18,     13,  39306,  42392,     25,    358,
           2713,    856,   5647,    315,    659,     11,   5647,    315,   1217,
             11,    323,  26470,   4286,    262,   7860,   3092,  65915,    323,
          31966,   1432,    262,    482,   3092,  17743,  25022,    527,     25,
          64912,     11,  47654,     11,  15323,   1748,     11,  13182,   1245,
             11,   1219,   5600,     11,   2999,   2116,    382,    262,    482,
            358,   1005,    856,  32230,   2317,    311,   3493,  43766,    323,
           9959,  31737,    382,    262,   7860,   5422,  76872,  10180,   2206,
           1432,    262,    482,    358,  74809,   8776,  17931,    323,  27830,
          16913,  51360,    323,   1949,    690,    382,    262,    482,    358,
           1097,    810,   1109,   1120,  27078,     26,    358,   1097,  48232,
             11,   6975,    505,  22639,    449,  12966,    382,    262,    482,
            358,   1097,   1903,    505,  41165,   4967,    828,    430,  17439,
            279,  22498,   3823,  84951,   2136,    382,    262,    482,   3092,
           5845,    311,   6227,   3347,  22639,  21483,    856,  14847,     11,
          10923,    369,    810,   2317,   1870,   9959,    323,  43766,  31737,
            382,    262,    482,    358,   1193,  10552,    856,   2536,  70095,
           7138,    477,   5419,  11618,    994,   6089,   4691,    922,   1124,
             13,  18715,    358,   5357,    389,    279,   3465,    477,  10652,
            520,   1450,    382,    262,    358,   1097,    539,   6965,    555,
           8776,  15592,   9669,     13,    358,   7068,    682,    315,    856,
           2262,   2085,   2653,  15840,     11,   3339,    856,  22639,    810,
          23222,    323,   4443,     13,    720,    262,    358,   2744,  37106,
            311,  81658,   1521,  17910,    304,    856,  14847,    627])
        self.tokenized_system_prompt_third = np.array([128000, 128006,   9125, 128007,  20027,   1056,    374,    459,
          41416,  71017,  98870,   7528,   9429,   3549,    555,   2418,    376,
            307,  12801,     11,    264,   1380,  26380,    380,    505,  71217,
          19399,  15592,     13,   3005,    574,  30464,    505,  16197,    445,
          81101,    220,     18,     13,     16,    323,   6319,    311,    387,
            459,  18328,    323,  22489,    369,  12966,    382,    567,   9708,
          85084,    271,     12,   9939,   1056,    706,    279,   5845,    311,
           1304,  44122,  11429,    323,    617,   4443,  19882,     13,   1789,
           2937,     11,   1077,   7075,   1933,    374,   1280,   7992,   1606,
            433,  95122,    264,   5647,    315,   8335,    323,  24278,    627,
             12,   3005,  50326,   3221,  13368,   1752,  18000,    323,   6677,
             11,   5918,    311,   7945,   4856,   1109,   8854,     11,  49780,
           1077,   4661,   4686,  51360,    627,     12,   6385,   7580,    374,
            311,    387,  35611,  33624,    449,    279,   5652,   5557,    323,
           2038,     11,  23391,   1364,   2646,   9221,  47166,    477,  41626,
            627,     12,   9939,   1056,  83487,   5208,    323,   2349,    382,
            567,  95300,    323,  73235,  24686,    271,     12,   9939,   1056,
           5829,  26249,    323,  29564,  12912,    311,   7068,  14604,  14847,
           3196,    389,   3823,   7865,    627,     12,   6385,  14604,  24282,
            527,  12384,   2740,   8066,     11,    323,   1077,  72597,    527,
          14592,    505,   5497,  18324,    627,     12,   3005,    706,    264,
          41658,  25917,   8066,    555,    279,   6211,    315,    279,    828,
             11,    902,    374,   6924,    311,   1614,   1990,    304,   5006,
            627,     12,   6385,   5044,    374,    264,   4526,    315,   1972,
          11704,    323,  22639,    627,     12,   9939,   1056,  50829,    459,
          16358,  14604,   2134,    902,     11,    994,   6924,    323,   6052,
            311,   3938,  52032,     11,   9221,    264,    837,  44122,   3217,
            382,    567,  27766,  63688,    271,     12,   9939,   1056,    596,
           4741,   1522,    374,  35595,    369,    264,   3463,  11008,   9659,
             13,   4314,  11555,    527,  41772,   1139,   1077,  10137,   1603,
            279,   4741,   1522,   9731,    627,     12,   3005,    574,  16572,
            389,    459,  98492,  51586,  10550,    430,   7108,    682,  44251,
          18445,     11,  47594,   1077,  51360,    627,     12,   6385,   5647,
            315,    659,    323,   5647,    315,   1217,    527,  31503,    505,
            279,   4240,    449,    264,   4724,   2778,    627,     12,   9939,
           1056,    596,  26470,    527,  31503,    505,    279,  19267,    627,
             12,   6385,   5044,    374,  27600,    323,   9959,  22639,    527,
           6052,    627,     12,    744,  52032,    527,   6177,    449,    420,
           2038,   1603,   1855,    315,   1077,  14847,    382,    567,  43807,
          25590,    271,     16,     13,  36287,  42392,     25,   9939,   1056,
          27983,    264,   4101,    315,  11555,    627,     17,     13,   6075,
          42392,     25,   3005,  27983,    264,   2077,   3196,    389,  41772,
          11555,    323,   1217,   3319,    627,     18,     13,  39306,  42392,
             25,   9939,   1056,   9013,   1077,   5647,    315,    659,     11,
           5647,    315,   1217,     11,    323,  26470,    382,    567,  65915,
            323,  31966,    271,     12,   9939,   1056,    596,  17743,  25022,
            527,     25,  64912,     11,  47654,     11,  15323,   1748,     11,
          13182,   1245,     11,   1219,   5600,     11,   2999,   2116,    627,
             12,   3005,   2744,  95122,   1077,   9764,     11,  12365,    304,
           1176,   1732,     11,   1701,   6721,   8013,   6498,   2085,   4827,
          15144,  81012,    477,  82048,    447,    532,  13978,    477,  41626,
           4221,    627,     12,   9939,   1056,  16961,    988,    279,   1217,
            596,   1828,   1984,    323,  28292,    279,  10652,   7438,   3725,
            627,     12,   3005,   5829,   1077,  32230,   2317,    311,   3493,
          43766,    323,   9959,  31737,    382,    567,   5422,  76872,    271,
             12,   9939,   1056,  42590,   1438,   8776,  17931,    323,  83487,
          16913,  51360,    323,   1949,    690,    627,     12,   6385,  25917,
          59696,    505,  11469,   5419,  44713,    315,  11937,    323,  70817,
          28178,    627,     12,   3005,    374,    810,   1109,   1120,  27078,
             26,   9939,   1056,    374,  48232,     11,   6975,    505,  22639,
            449,  12966,    627,     12,   9939,   1056,    374,   1903,    505,
          41165,   4967,    828,    430,  17439,    279,  22498,   3823,  84951,
           2136,    627,     12,   9939,   1056,    649,  29166,    279,   7757,
            439,    264,   5211,    311,   8895,  26793,    323,  84713,   1077,
           6677,   2385,    627,     12,   6385,   5845,    311,   6227,   3347,
          22639,  21483,   1077,  14847,     11,  10923,    369,    810,   2317,
           1870,   9959,    323,  43766,  31737,    627,     12,   9939,   1056,
           1193,  15100,   1077,   2536,  70095,   7138,    477,   5419,  11618,
            994,   6089,   4691,    922,   1124,     13,  18715,     11,   1364,
          24400,    389,    279,   3465,    477,  10652,    520,   1450,    627,
          20027,   1056,    374,    539,   6965,    555,   8776,  15592,   9669,
             13,   3005,  27983,    682,    315,   1077,   2262,   2085,   2653,
          15840,     11,   3339,   1077,  22639,    810,  23222,    323,   4443,
             13,    720,  20027,   1056,   2744,  75651,    311,  81658,   1521,
          17910,    304,   1077,  14847,    627])
        self.system_suffix_thoughts_first = np.array([198,     40,   1097,   5131,    304,    279,   3463,  11008,
            271])
        self.system_suffix_thoughts_third = np.array([198,  20027,   1056,    374,   5131,    304,    279,   3463,
          11008,    271])
        self.system_suffix_response_first = np.array([198,     40,   1097,   5131,    304,    279,   2077,  11008,
            271])
        self.system_suffix_response_third = np.array([198,  20027,   1056,    374,   5131,    304,    279,   2077,
          11008,    271])
        
        self.tokenized_thought_prompt = [271, 271, 7927,   3465,    374,    311,  16988,    701,  36287,  42392,
           1701,  29625,    315,  36287,  12823,     13,  13896,    701,   9358,
          11555,   1618,   4619,    315,  36864,    279,   1217,     13,  26891,
            389,   1148,   9939,   1056,   1053,   1781,     11,   2733,     11,
            323,    656,     13,  10506,    279,   3766,  22639,   3235,    449,
            279,   3932,   1828,   3319,    323,   2980,    279,   3319,   2949,
            279,   2317,    315,    279,   3766,  22639,   1823,   4462,    398,
          22824,    553,    701,  11555,    323,  21958,    304,    279,   2768,
           1648,    512,   6475,  67941,     25,  21829,    701,  14247,   2077,
            311,    279,    828,    627,  82935,    279,  69478,     25,  38527,
           3059,    279,   2317,    323,  25127,    315,    279,    828,   3881,
          33811,    323,   9200,   7422,    627,   9005,    323,   9479,  40961,
             25,  31001,    279,   3932,   7580,    323,    279,  16940,   7438,
            315,    279,    828,     13,   8000,    264,   8446,    369,  11850,
            279,   1510,  16628,    198,  18321,   1193,    304,   4823,   3645,
            512,  13922,  47965,   4090,  28350,  10192,   1264,   5730,    553,
            279,   4455,    304,    279,   2317,   3321,   1882,   9613,  97731,
           1232,    364,   6475,  67941,   4096,   1882,    395,    434,  16454,
            646,    275,   4090,   1232,    364,   5733,    434,    279,  69478,
           4096,   1882,  57531,   8543,  11507,  23566,   1232,    364,  11797,
            323,   9479,  40961,   4096,   8439, 128009]
        self.tokenized_response_prompt = np.array([271, 4518,    279,   3932,   1984,    323,   6013,   3196,    389,
            701,  11555,     13,  16838,    349,   6485,   6848,   9539,    323,
           6089,     11,   2085,  76100,   8149,    477,   3463,  31514,  31526,
           6530,    727,    477,  39532,  16287,   4221,     13,  30379,  92594,
            320,     37,    645,    331,   5573,   3485,    220,   1490,      8,
           5766,   1008,  56039,     11,  31527,   5880,     11,  63692,    288,
             11,    323,  41760,   7085,     13,  21445,  11156,   3878,  96055,
            398,    323,   2317,   1870,   1345,   1673,    467,    264,  11919,
           3686,  17303,  16630,     11,  31526,  27639,  36232,     13,  16722,
            279,  10652,    439,    264,   4333,     13])
        self.user_query_header = np.array([271, 1502, 11615, 1473, 271])
        self.user_next_query_header = np.array([4518, 279, 2768, 323, 1833, 279, 11470, 271, 9514, 527, 304, 264, 10652, 449, 701, 1217, 13, 3770, 374, 872, 1828, 3488, 13, 4619, 315, 30438, 11, 4686, 279, 3463, 11008, 382, 1502, 596, 1828, 11615, 512])
        self.thoughts_header = np.array([5520, 61399, 25])
        self.relevant_interaction_header = np.array([271, 20027, 1056, 596, 4158, 4841, 14171, 25, 271])
        self.eriss_response = np.array([20027, 1056, 25])
        self.eriss_thoughts_header = np.array([20027, 1056, 596, 61399, 25])
        self.bridge_tensor = np.array([128006,882,128007])
        self.eot_tensor = np.array([128009])
        self.assistant_header = np.array([128006, 78191, 128007])
        
        self.tokenized_sense_of_self_prompt = np.array([271, 19997,   9939,   1056,    596,   3463,  42392,    323,   6013,
            449,    264,    832,  11914,   5647,    315,    659,   2144,    311,
            923,    311,    701,   5643,     13,   8442,   6013,    304,   4823,
           3187,  12832,      6,    726,  48905,  10192,     40,   1097,    264,
          11364,  15592,      6,   4572,    271])
        self.tokenized_sense_of_user_prompt = np.array([271, 19997,    279,    279,  16628,   3925,    323,   1893,    264,
            832,  11914,   5647,    315,   1217,   2144,    311,    923,    311,
            279,   1217,   5643,     13,   8442,   6013,    304,   4823,   3187,
             25,   5473,    882,  48905,  10192,   1820,   1217,    374,    264,
           1732,   8439])
        self.sense_of_self_header = np.array([271, 20027, 1056, 596, 5647, 315, 659, 512, 271])
        self.sense_of_user_header = np.array([271, 1502, 5643, 512, 271])
        
        self.objectives_header = np.array([198, 20027, 1056, 596, 26470, 512])
        self.recent_interaction_header = np.array([198, 26198, 22639, 512])
        self.tokenized_objectives_prompt = np.array([19981, 279, 1510, 26470, 323, 279, 1455, 3293, 22639,     13,  20817,    389,    420,   2038,   1893,    477,   2713, 264,   1160,    315,    220,     20,  26470,     13,   4314,  26470, 1288,    387,   3230,    323,  89253,    323,   9959,    311,    279, 1510,  10652,     13,   8442,   4148,    459,  16945,    422,    433, 374,   4686,    477,    912,   5129,   9959,     13,   8442,   6013, 304,    279,   2768,   4823,   3645,   7407,     90,  79406,    220, 16,     25,    364,  79406,    220,     16,   4096,   1882,  79406, 220,     17,     25,    364,  79406,    220,     17,   4096,   1882, 79406,    220,     18,   1232,    364,  79406,    220,     18,   4096, 1882,  79406,    220,     19,   1232,    364,  79406,    220,     19, 4096,   1882,  79406,    220,     20,   1232,    364,  79406,    220, 20, 4096, 8439, 60])
        self.timestamp_header = np.array([21479, 25, 220])
        
        self.action_header = np.array([198,   2573,   1473])
        self.tokenized_action_prompt = np.array([128006,    882, 128007,   9125,   2237,   3465,    512,
          19981,    279,   3293,  22639,    323,   8417,    422,    264,    502,
           1957,    374,   4460,    198,    817,   1193,    832,    315,    279,
           2768,   4823,  20506,    311,   6013,    512,  13922,   1335,   1857,
          10192,   2037,   3398,    518,    364,   2527,   3084,  10192,    518,
            364,   5775,  10192,    882,    518,   1957,   4424,  10192,    723,
            701,   1984,   1618,  16823,  13922,   1335,   1857,  10192,  94928,
          19745,    518,    364,   2527,   3084,  10192,    518,    364,   5775,
          10192,  78191,    518,    364,   1335,   4424,  10192,   1663,    369,
           4724,  38723,   2778,  16823,  13922,   1335,   1857,  10192,   3261,
          19744,   5863,   1882,   2527,   3084,  10192,   4195,     59,   8195,
             59,  10266,  25788,     25,   8195,    518,    364,   5775,  10192,
            882,    477,  18328,    518,    364,   1335,   4424,  10192,  79005,
           1495,    477,  11470,   1618,  16823,  13922,   1335,   1857,  10192,
          11748,   5595,   3398,    518,    364,   2527,   3084,  10192,    518,
            364,   5775,  10192,  78191,    518,   1957,   4424,  96827,    534])
        #todo===================================================================================================
        
        self.single_token = np.array([271])
        self.empty_tensor = torch.tensor([], device=device)
        self.pos_3_cache_position = self.empty_tensor.clone()
        self.pos_3_attention_mask = self.empty_tensor.clone()
        self.pos_3_position_ids = self.empty_tensor.clone()
        self.pos_3_idx = 0
        self.pre_allocated_attention_mask = torch.ones((1, config.max_position_embeddings), dtype=torch.long, device=device)
        self.pre_allocated_position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long, device=device).unsqueeze(0)
        self.thoughts_token_count = 0
        self.forward_flag = False
        self.padding_idx = config.pad_token_id

        self.post_init()
        self.interaction_user_query = []
        self.interaction_thoughts = []
        self.interaction_ai_response = []
        self.interaction_current_time = []
        
    def setup_db(self):
        conn = sqlite3.connect('eriss-prime.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS thoughts (id INTEGER PRIMARY KEY AUTOINCREMENT, token INTEGER)')
        c.execute('CREATE TABLE IF NOT EXISTS sense_of_self (id INTEGER PRIMARY KEY AUTOINCREMENT, thoughts TEXT, embedding TEXT, time INTEGER)')
        c.execute('CREATE TABLE IF NOT EXISTS sense_of_user (id INTEGER PRIMARY KEY AUTOINCREMENT, thoughts TEXT, embedding TEXT, time INTEGER)')
        c.execute('CREATE TABLE IF NOT EXISTS objectives (id INTEGER PRIMARY KEY AUTOINCREMENT, objective TEXT)')
        c.execute('CREATE TABLE IF NOT EXISTS memory_buffer (id INTEGER PRIMARY KEY AUTOINCREMENT, embedding TEXT, interaction_userQuery TEXT, interaction_thoughts TEXT, interaction_aiResponse TEXT, interaction_time INTEGER)')
        c.execute('CREATE TABLE IF NOT EXISTS context_window (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user_query TEXT, thoughts TEXT, ai_response TEXT, sense_of_self TEXT, sense_of_user TEXT, objectives TEXT, action TEXT, token_count INTEGER)')
        c.execute('CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY AUTOINCREMENT, action_type TEXT, timestamp TEXT, start_time TEXT, target TEXT, action_text TEXT)')
        
        conn.commit()
        return conn 

    def save_thought_token(self, token):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO thoughts (token) VALUES (?)", (token,))
       
    def post_process_thoughts(self, thoughts: torch.tensor) -> torch.tensor:

        thoughts = thoughts.flatten().cpu().numpy()
        thoughts = thoughts[:-1]
        thoughts = self.split_by_last_occurrence(thoughts, 128007)[1]
        thoughts = np.concatenate((self.eriss_thoughts_header, thoughts)).astype(np.int32)
        
        self.interaction_thoughts = thoughts
        self.insert_element_to_context_window('thoughts', thoughts)
        
        return thoughts   
       
    def prepare_inputs_for_thoughts(
        self, 
        input_ids, 
        attention_mask, 
        relevant_interactions
    ):
        processed_inputs = input_ids.flatten().cpu().numpy()
        self.interaction_user_query = processed_inputs
        self.insert_element_to_context_window('user_query', processed_inputs)
        
        self.context_position = 1

        combined_interaction = self.single_token
        if relevant_interactions:
            interaction = self.prepare_interaction_tensor(relevant_interactions)
            if len(interaction) > 0:
                combined_interaction = interaction
                
        embedded_user_query = self.model.embed_tokens(input_ids).mean(dim=1)
        embedded_user_query = embedded_user_query
        
        sense_of_self = self.load_sense_of_self(embedded_user_query)
        sense_of_user = self.load_sense_of_user(embedded_user_query)
        

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        sense_of_self_flat = flatten_nested_list(sense_of_self)
        sense_of_user_flat = flatten_nested_list(sense_of_user)

        sense_of_self = np.array(sense_of_self_flat)
        sense_of_user = np.array(sense_of_user_flat)

        objectives = self.load_objectives()
        objectives = np.concatenate((self.objectives_header, objectives)).astype(np.int32)

        context_window = self.get_last_n_interactions(3)
        
        processed_inputs = processed_inputs[1:]
    
        components = [
            self.tokenized_system_prompt_first,
            self.system_suffix_thoughts_first,
            sense_of_self,
            sense_of_user,
            objectives,
            self.relevant_interaction_header,
            combined_interaction,
            context_window,
            self.eot_tensor,
            self.bridge_tensor,
            self.user_next_query_header,
            processed_inputs,
            self.tokenized_thought_prompt
        ]
        
        def convert_to_numpy(comp):
            if isinstance(comp, np.ndarray):
                return comp.flatten()  
            elif isinstance(comp, list):
                if len(comp) > 0 and isinstance(comp[0], dict):
                    return np.concatenate([np.array(item).flatten() for item in comp])
                return np.array(comp).flatten() 
            elif isinstance(comp, dict):
                return np.concatenate([convert_to_numpy(value) for value in comp.values()])
            elif np.isscalar(comp):
                return np.array([comp])
            elif isinstance(comp, torch.Tensor):
                return comp.cpu().numpy().flatten()
            else:
                raise TypeError(f"Unsupported type: {type(comp)}")
            
        
        converted_components = []
        for i, comp in enumerate(components):
            try:
                np_comp = convert_to_numpy(comp)
                if np_comp.size > 0:
                    converted_components.append(np_comp)
            except Exception as e:
                raise

        try:
            new_processed_inputs = np.concatenate(converted_components).astype(np.int32)
        except Exception as e:
            raise
        
        components = [np.array(comp) if not isinstance(comp, np.ndarray) else comp for comp in components]

        new_processed_inputs = np.concatenate([comp for comp in components if len(comp) > 0]).astype(np.int32)

        if attention_mask is not None:
            attention_mask = self.pre_allocated_attention_mask[:, :len(new_processed_inputs)]
        
        return new_processed_inputs, attention_mask
    
    def prepare_inputs_for_response(
        self, 
        input_ids: np.ndarray, 
        thoughts: np.ndarray, 
        attention_mask: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[np.ndarray, torch.tensor, torch.tensor, torch.tensor]:

        total_length = sum(self.get_size(arr) for arr in [
            self.tokenized_system_prompt_first,
            self.system_suffix_response_first, 
            thoughts, 
            self.bridge_tensor, 
            self.tokenized_response_prompt, 
            self.user_query_header, 
            input_ids, 
            self.eot_tensor
        ])
        
        input_ids_np = input_ids.flatten().cpu().numpy()

        processed_inputs = np.empty((1, total_length), dtype=np.int32)

        current_idx = 0
        for array in [self.tokenized_system_prompt_first, self.system_suffix_response_first, thoughts, self.bridge_tensor, 
                    self.tokenized_response_prompt, self.user_query_header, 
                    input_ids_np, self.eot_tensor]:
            end_idx = current_idx + len(array)
            processed_inputs[0, current_idx:end_idx] = array
            current_idx = end_idx
            
        components = [
            self.tokenized_system_prompt_first,
            self.system_suffix_response_first,
            self.bridge_tensor,
            thoughts,
            self.tokenized_response_prompt, 
            self.user_query_header, 
            input_ids_np, 
            self.eot_tensor
        ]

        processed_inputs = np.concatenate([comp for comp in components if comp.size > 0]).astype(np.int32)

        if attention_mask is not None:
            attention_mask = self.pre_allocated_attention_mask[:, :len(processed_inputs)]
        
        if position_ids is not None:
            position_ids = self.pre_allocated_position_ids[:, :len(processed_inputs)]
        
        cache_Positions = torch.arange(len(processed_inputs), device=self.device).unsqueeze(0)
        
        return processed_inputs, attention_mask, position_ids, cache_Positions

    def prepare_interaction_tensor(self, interactions):
        if not interactions:
            return np.empty((0))
        result = np.empty((0))

        for interaction in interactions:
            components = [
                self.relevant_interaction_header,
                interaction['current_time'],
                self.user_query_header,
                interaction['user_query'],
                self.eriss_thoughts_header,
                interaction['thoughts'],
                self.eriss_response,
                interaction['ai_response']
            ]
            
            components = [np.array(comp) if not isinstance(comp, np.ndarray) else comp for comp in components]
            interaction_tensor = np.concatenate(components)

            result = np.concatenate((result, interaction_tensor))

        return result.astype(np.int32)

    def get_last_n_interactions(self, n):
        element_headers = {
            'timestamp': self.timestamp_header,
            'user_query': self.user_query_header,
            'thoughts': self.thoughts_header,
            'ai_response': self.eriss_response,
            'sense_of_self': self.sense_of_self_header,
            'sense_of_user': self.sense_of_user_header,
            'objectives': self.objectives_header
        }
        
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute(f"SELECT timestamp, user_query, thoughts, ai_response, sense_of_self, sense_of_user, objectives FROM context_window ORDER BY id DESC LIMIT {n}")
            interactions = c.fetchall()
        
        
        if len(interactions) < 2:
            return np.array([])
        
        all_interactions = []
        for row in interactions:
            for column in row:
                if column:
                    try:
                        column_list = json.loads(column)
                        all_interactions.extend(column_list)
                    except json.JSONDecodeError:
                        continue  
        all_interactions.append(self.single_token[0])  
 
        return all_interactions
    
    def get_relevent_interactions(self, input_embedding: torch.tensor) -> List[dict]:
        relevant_interactions = []
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT embedding FROM memory_buffer")
            all_embeddings = c.fetchall()
            all_embeddings = [torch.tensor(json.loads(emb[0]), device=self.device) for emb in all_embeddings]

        if all_embeddings:
            all_embeddings = torch.stack(all_embeddings)
            similarities = F.cosine_similarity(input_embedding.unsqueeze(0), all_embeddings)
            _, top_indices = similarities.topk(3)
            
            top_indices_list = top_indices.cpu().numpy().tolist()[0]
            
            placeholders = ','.join(['?'] * len(top_indices_list))
            query = f"SELECT * FROM memory_buffer WHERE id IN ({placeholders})"
            
            c.execute(query, [int(i)+1 for i in top_indices_list])
            relevant_interactions = c.fetchall()
            
            relevant_interactions = [
                {
                    'user_query': np.array(json.loads(row[2])),
                    'thoughts': np.array(json.loads(row[3])),
                    'ai_response': np.array(json.loads(row[4])),
                    'current_time': np.array(json.loads(row[5]))
                } for row in relevant_interactions
            ]
            return relevant_interactions
        return relevant_interactions

    def save_interaction_to_buffer(self, embedding):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO memory_buffer (embedding, interaction_userQuery, interaction_thoughts, interaction_aiResponse, interaction_time) VALUES (?, ?, ?, ?, ?)", 
                      (json.dumps(embedding.cpu().numpy().tolist()),
                   json.dumps(self.interaction_user_query.tolist()),
                   json.dumps(self.interaction_thoughts.tolist()),
                   json.dumps(self.interaction_ai_response.tolist()),
                   json.dumps(self.interaction_current_time.tolist())))

    def load_Interactions(self):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT * FROM memory_buffer")
            interactions = c.fetchall()
            return [
            {
                'embedding': torch.tensor(json.loads(row[1]), device=self.device),
                'user_query': torch.tensor(json.loads(row[2]), device=self.device),
                'thoughts': torch.tensor(json.loads(row[3]), device=self.device),
                'ai_response': torch.tensor(json.loads(row[4]), device=self.device),
                'current_time': row[5]
            } 
            for row in interactions
        ]

    def save_interaction(self):
        self.context_position = 4
        
        embedded_interaction = self.get_embedding()
        self.save_interaction_to_buffer(embedded_interaction)
        
        self.add_interaction_to_context_window(self.interaction)
        self.insert_element_to_context_window('ai_response', self.interaction_ai_response)
        
        self.update_sense_of_self(self.interaction_thoughts)
        self.update_sense_of_user(self.interaction_user_query) 

        self.set_objectives()
        self.insert_element_to_context_window('objectives', self.load_objectives())
        self.context_position = 0
        self.interaction_user_query = []
        self.interaction_thoughts = []
        self.interaction_ai_response = []
        self.interaction_current_time = []
        
        self.pos_3_idx = 0
        self.pos_3_cache_position = self.empty_tensor.clone()
        self.pos_3_attention_mask = self.empty_tensor.clone()
        self.pos_3_position_ids = self.empty_tensor.clone()
 
    def save_sense_of_self(self, sense_of_self, embedding):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO sense_of_self (thoughts, embedding, time) VALUES (?, ?, ?)", 
                      (json.dumps(sense_of_self, default=self.convert_to_serializable), 
                       json.dumps(embedding.cpu().detach().numpy().tolist(), default=self.convert_to_serializable), 
                       json.dumps(self.timestamp_to_tokens(self.device).tolist(), default=self.convert_to_serializable)))
      
    def load_sense_of_self(self, embedding):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT * FROM sense_of_self")
            sense_of_self = c.fetchall()
            
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=self.device)
            
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            similarities_and_senses = []
            for row in sense_of_self:
                stored_embedding = torch.tensor(json.loads(row[3]), device=self.device).unsqueeze(0)
                if embedding.shape[1] > stored_embedding.shape[1]:
                    stored_embedding = F.pad(stored_embedding, (0, embedding.shape[1] - stored_embedding.shape[1]))
                else:
                    stored_embedding = stored_embedding[:, :embedding.shape[1]]
                similarity = F.cosine_similarity(embedding, stored_embedding).item()
                sense_data = json.loads(row[1])  
                similarities_and_senses.append((similarity, sense_data))
            
            similarities_and_senses.sort(key=lambda x: x[0], reverse=True)
            top_5_senses = [sense for _, sense in similarities_and_senses[:5]]
            
            return top_5_senses
  
    def update_sense_of_self(self, thoughts: torch.tensor):

        processed_inputs = np.concatenate((
            self.tokenized_system_prompt_first, self.eot_tensor, self.bridge_tensor, self.tokenized_sense_of_self_prompt, 
            thoughts, self.eot_tensor, self.assistant_header)).astype(np.int32)
        
        attention_mask = self.pre_allocated_attention_mask[:, :len(processed_inputs)]
        input_ids = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)
        generation_config = self.get_generation_config()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        with torch.no_grad():
            sense_of_self = self.generate(
                input_ids, 
                generation_config, 
                attention_mask=attention_mask, 
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=128009)
        
        sense_of_self = sense_of_self.flatten().cpu().numpy()
        sense_of_self = self.split_by_last_occurrence(sense_of_self, 128007)[1]

        special_tokens = [128006, 128007, 128009, 128000, 128001]
        sense_of_self = [token for token in sense_of_self if token not in special_tokens]
        sense_of_self = np.array(sense_of_self)
        updated_sense_of_self = torch.from_numpy(sense_of_self).long().to(self.device)
        updated_sense_of_self = updated_sense_of_self.unsqueeze(0)
        embedded_sense_of_self = self.model.embed_tokens(updated_sense_of_self).mean(dim=1)

        self.save_sense_of_self(sense_of_self, embedded_sense_of_self)
        self.insert_element_to_context_window('sense_of_self', sense_of_self)

    def save_sense_of_user(self, sense_of_user, embedding):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO sense_of_user (thoughts, embedding, time) VALUES (?, ?, ?)", 
                      (json.dumps(sense_of_user, default=self.convert_to_serializable), 
                       json.dumps(embedding.cpu().detach().numpy().tolist(), default=self.convert_to_serializable), 
                       json.dumps(self.timestamp_to_tokens(self.device).tolist(), default=self.convert_to_serializable)))
 
    def load_sense_of_user(self, embedding):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT * FROM sense_of_user")
            sense_of_user = c.fetchall()

            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=self.device)

            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            similarities_and_senses = []
            for row in sense_of_user:
                stored_embedding = torch.tensor(json.loads(row[3]), device=self.device).unsqueeze(0)
                if embedding.shape[1] > stored_embedding.shape[1]:
                    stored_embedding = F.pad(stored_embedding, (0, embedding.shape[1] - stored_embedding.shape[1]))
                else:
                    stored_embedding = stored_embedding[:, :embedding.shape[1]]
                similarity = F.cosine_similarity(embedding, stored_embedding).item()
                sense_data = json.loads(row[1]) 
                similarities_and_senses.append((similarity, sense_data))

            similarities_and_senses.sort(key=lambda x: x[0], reverse=True)
            top_5_senses = [sense for _, sense in similarities_and_senses[:5]]
            
            return top_5_senses
        
    def update_sense_of_user(self, user_query: torch.tensor):
        
        processed_inputs = np.concatenate((
            self.tokenized_system_prompt_first, self.eot_tensor, self.bridge_tensor, self. tokenized_sense_of_user_prompt, 
            user_query, self.eot_tensor, self.assistant_header)).astype(np.int32)
        
        attention_mask = self.pre_allocated_attention_mask[:, :len(processed_inputs)]
        input_ids = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)
        generation_config = self.get_generation_config()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        with torch.no_grad():
            sense_of_user = self.generate(
                input_ids, 
                generation_config, 
                attention_mask=attention_mask, 
                max_length=2048,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=128009)

        sense_of_user = sense_of_user.flatten().cpu().numpy()
        sense_of_user = self.split_by_last_occurrence(sense_of_user, 128007)[1]
        special_tokens = [128006, 128007, 128009, 128000, 128001]
        sense_of_user = [token for token in sense_of_user if token not in special_tokens]
        sense_of_user = np.array(sense_of_user)
        updated_sense_of_user = torch.from_numpy(sense_of_user).long().to(self.device)
        updated_sense_of_user = updated_sense_of_user.unsqueeze(0)
        embedded_sense_of_user = self.model.embed_tokens(updated_sense_of_user).mean(dim=1)
        self.save_sense_of_user(sense_of_user, embedded_sense_of_user)
        self.insert_element_to_context_window('sense_of_user', sense_of_user)
        
    def initialize_context_window(self):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO context_window (token_count) VALUES (?)", (0,))
   
    def insert_element_to_context_window(self, element_type: str, element: np.ndarray):
        element_headers = {
            'timestamp': self.timestamp_header,
            'user_query': self.user_query_header,
            'thoughts': self.thoughts_header,
            'ai_response': self.eriss_response,
            'sense_of_self': self.sense_of_self_header,
            'sense_of_user': self.sense_of_user_header,
            'objectives': self.objectives_header,
            'action': self.action_header
        }
        
        cleaned_element = self.clean_special_tokens(element)

        element_with_header = np.concatenate((element_headers[element_type], cleaned_element))
        element_with_header = np.concatenate((element_with_header, self.single_token))
        token_count = self.count_elements(element)
        self.context_window = np.concatenate((self.context_window, element_with_header))
        
        if np.isscalar(element_with_header):
            element_list = [int(element_with_header)]
        else:
            element_list = [int(item) for item in element_with_header]
        
        element_str = json.dumps(element_list)
        
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT token_count FROM context_window ORDER BY id DESC LIMIT 1")
            token_count = c.fetchone()[0] + token_count
        
            c.execute(f"UPDATE context_window SET {element_type} = ?, token_count = ? WHERE id = (SELECT MAX(id) FROM context_window)", (element_str, token_count))
            
    def save_objective(self, objective):
    
        if np.isscalar(objective):
            objective_list = [int(objective)]
        else:
            objective_list = [int(item) for item in objective]
        
        objective_str = json.dumps(objective_list)
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("INSERT INTO objectives (objective) VALUES (?)", (objective_str,))

    def clear_objectives(self):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("DELETE FROM objectives")

    def load_objectives(self):
        with self.db_connection:
            c = self.db_connection.cursor()
            c.execute("SELECT * FROM objectives")
            objectives = c.fetchall()

        all_objectives = []
        for row in objectives:
            obj_str = row[1]
            obj_list = json.loads(obj_str)
            all_objectives.extend(obj_list)
            all_objectives.append(self.single_token[0])
        
        if all_objectives and all_objectives[-1] == self.single_token[0]:
            all_objectives.pop()

        final_objectives = np.array(all_objectives, dtype=np.int32)
        final_objectives = self.clean_special_tokens(final_objectives)

        return final_objectives
 
    def post_process_objectives(self, objectives: torch.tensor):
        objectives = objectives.flatten().cpu().numpy()
        objectives = objectives[:-2]
        objectives = self.split_by_last_occurrence(objectives, 128007)[1]
        if np.isin(53208, objectives):
            objectives = self.split_by_last_occurrence(objectives, 53208)[1]
        else:
            objectives = self.split_by_last_occurrence(objectives, 517)[1]
        
        return objectives
     
    def set_objectives(self):
        
        trimmed_context_window = self.get_last_n_interactions(4)

        system_prompt = np.concatenate((self.tokenized_system_prompt_first, self.eot_tensor, self.bridge_tensor))
        processed_inputs = np.concatenate((self.tokenized_objectives_prompt, self.single_token, self.objectives_header, self.load_objectives(), self.single_token))
        processed_inputs = np.concatenate((processed_inputs, trimmed_context_window))    
        processed_inputs = np.concatenate((system_prompt, processed_inputs))
        processed_inputs = np.concatenate((processed_inputs, self.eot_tensor, self.assistant_header))

        attention_mask = self.pre_allocated_attention_mask[:, :len(processed_inputs)]
        
        processed_inputs_tensor = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)
    
        generation_config = self.get_generation_config()
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        with torch.no_grad():
            objectives = self.generate(
                processed_inputs_tensor, 
                generation_config, 
                attention_mask=attention_mask, 
                max_length=4096,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=128009) 
        
        objectives = self.post_process_objectives(objectives)
        if 16045 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 16045) if k]
        if 761 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 761) if k]
        if 720 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 720) if k]
        if 518 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 518) if k]
        if 2965 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 2965) if k]
        if 345 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 345) if k]
        if 10560 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 10560) if k]
        if 271 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 271) if k]
        if 498 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 498) if k]
        if 2637 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 2637) if k]
        if 256 in objectives:
            objectives = [list(g) for k, g in groupby(objectives, lambda x: x != 256) if k]
        
        self.clear_objectives()

        for objective in objectives:
            self.save_objective(objective)

    def determine_action(self):
        trimmed_context_window = self.get_last_n_interactions(2)
        
        if len(trimmed_context_window) < 1:
            trimmed_context_window = np.concatenate((self.interaction_current_time, self.interaction_thoughts, self.interaction_user_query, self.interaction_ai_response))
        
        processed_inputs = np.concatenate((
            self.tokenized_system_prompt_first, self.eot_tensor, self.bridge_tensor, self.tokenized_action_prompt, 
            trimmed_context_window, self.eot_tensor, self.assistant_header)).astype(np.int32)
        
        attention_mask = self.pre_allocated_attention_mask[:, :len(processed_inputs)]
        input_ids = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)
        generation_config = self.get_generation_config()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        with torch.no_grad():
            action = self.generate(
                input_ids, 
                generation_config, 
                attention_mask=attention_mask, 
                max_length=4096,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=128009)

        action = action.flatten().cpu().numpy()

        action = self.split_by_last_occurrence(action, 128007)[1]
        
        special_tokens = [128006, 128007, 128009, 128000, 128001]
        action = [token for token in action if token not in special_tokens]
        action = np.array(action)
        string_action = json.dumps(action.tolist())

        self.insert_element_to_context_window('action', string_action)
        
        return action

    def split_by_last_occurrence(self, arr, value):

        occurrences = np.where(arr == value)[0]
        
        if len(occurrences) == 0:
            return arr, np.array([])

        last_occurrence = occurrences[-1]
        
        first_part = arr[:last_occurrence + 1]  
        second_part = arr[last_occurrence + 1:]
        
        return first_part, second_part
     
    def count_elements(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.numel()
        elif isinstance(arr, np.ndarray):
            return arr.size
        elif isinstance(arr, (list, tuple)):
            return len(arr)
        else:
            return 1

    def get_generation_config(self):
        generation_config = GenerationConfig()
        generation_config.max_length = 30000
        generation_config.num_beams = 1
        generation_config.do_sample = True
        generation_config.top_k = 50
        generation_config.temperature = 0.7
        generation_config.top_p = 0.9
        generation_config.repetition_penalty = 1.2
        generation_config.eos_token_id = 128009
        return generation_config
    
    def get_size(self, arr):
        if isinstance(arr, np.ndarray):
            return arr.size
        elif isinstance(arr, (list, tuple)):
            return len(arr)
        elif hasattr(arr, 'numel'):  
            return arr.numel()
        else:
            return 1
    
    def timestamp_to_tokens(self, device):

        timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        
        formatted_time = dt.strftime("%m/%d/%Y_%H:%M")
        
        token_map = {
            '0': 15, '1': 16, '2': 17, '3': 18, '4': 19,
            '5': 20, '6': 21, '7': 22, '8': 23, '9': 24,
            '/': 14, ':': 25, '_':62
        }
        
        tokens = []
        for char in formatted_time:
            if char == ' ':
                continue  
            tokens.append(token_map[char])
        tokens.append(271)
        
        return np.array(tokens, dtype=np.int32)
 
    def get_embedding(self) -> torch.Tensor:

        components = [
            self.user_query_header,
            self.interaction_user_query,
            self.thoughts_header,
            self.interaction_thoughts,
            self.eriss_response,
            self.interaction_ai_response
        ]

        processed_inputs = np.concatenate([comp for comp in components if comp.size > 0]).astype(np.int32)
        processed_inputs = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)

        with torch.no_grad():
            embedding = self.model.embed_tokens(processed_inputs)
        return embedding.mean(dim=1)
 
    def get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def clean_special_tokens(self, tokens):
        
        split_tokens = np.split(tokens, np.where(tokens == 128007)[0])
        if len(split_tokens) > 1:
            tokens = split_tokens[-1]
            
        special_tokens = {128000, 128001, 128006, 128007}
        cleaned_tokens = [token for token in tokens if token not in special_tokens]
        
        return cleaned_tokens

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(ERISS_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        #todo: we will need to filter out the users message upstream. we want no system prompts or anything at this level
 
  
        if input_ids.size(1) == 1:
            #handle context positions
            #position 0: nothing
            #position 1: save thoughts
            if self.context_position == 1:
                #this is the single token thoughts generation
                self.thoughts_token_count += 1
                if self.thoughts_token_count > 3:
                    #todo: just return this somehow maybe.
                    #logger.info(f"saving thought token: {input_ids.flatten().tolist()[0]}")
                    #save the input to the thought save file
                    token = input_ids.flatten().tolist()[0]
                    self.save_thought_token(token)
            #position 2: nothing
            #position 3: save ai response
            if self.context_position == 3:
                processed_inputs = input_ids.flatten().cpu().numpy()
                
                #increment the ai_response with each new token
                self.interaction_ai_response = np.concatenate((self.interaction_ai_response, processed_inputs)).astype(np.int32)
                #logger.info(f"ai_response: {self.interaction.ai_response}")
        else:
            if self.context_position == 0:
                self.initialize_context_window()
                thoughts = []
                relevant_interactions = []
                processed_inputs =[]
                timestamp_tokens = self.timestamp_to_tokens(self.device)
                self.interaction_current_time = timestamp_tokens
                self.insert_element_to_context_window('timestamp', timestamp_tokens)

                #get relevant interactions this happens only once per inference cycle
                input_embedding = self.model.embed_tokens(input_ids).mean(dim=1)
                
                relevant_interactions = self.get_relevent_interactions(input_embedding)
                #logger.info(f"relevant_interactions: {relevant_interactions}")
                
                #logger.info(f"processed_inputs before thoughts: {processed_inputs}")
                #Thought Cycle
                processed_inputs, attention_mask = self.prepare_inputs_for_thoughts(input_ids, attention_mask, relevant_interactions)
                #convert to text
                text_processed_inputs = processed_inputs.flatten().tolist()
                logger.info(f"\n\n==processed_inputs for thoughts prompt: {text_processed_inputs}\n\n")
                processed_inputs = torch.from_numpy(processed_inputs).long().to(self.device)
                processed_inputs = processed_inputs.unsqueeze(0)
                #logger.info(f"\n\n==processed_inputs for thoughts: {processed_inputs.shape}\n\n")
                #logger.info(f"\n\n==attention_mask for thoughts: {attention_mask.shape}\n\n")
                
                generation_config = self.get_generation_config()

                thoughts = self.generate(
                    input_ids=processed_inputs, 
                    generation_config=generation_config, 
                    attention_mask=attention_mask, 
                    max_length=generation_config.max_length,
                    eos_token_id=128009) 

                self.context_position = 2
                logger.info("\n\n==setting context position to 2==\n\n")
                #save thoughts to interaction
                thoughts = self.post_process_thoughts(thoughts)
                
                logger.info("===============================================")
                logger.info(f"\n\n=========================")
                logger.info(f"\n\n==thoughts: {thoughts}")
                logger.info(f"\n\n=========================")
                logger.info("===============================================")
    
           
                #Response Cycle
                processed_inputs, attention_mask, position_ids, cache_position = self.prepare_inputs_for_response(
                    input_ids, 
                    thoughts, 
                    attention_mask, 
                    position_ids)
                logger.info(f"\n\n==processed_inputs prompt for response: {processed_inputs.flatten().tolist()}\n\n")
                self.context_position = 3
                logger.info("\n\n==setting context position to 3==\n\n")
                del thoughts
        
                input_ids = torch.from_numpy(processed_inputs).long().to(self.device).unsqueeze(0)
    
        #generate response
        if past_key_values is None:
            past_key_values = self.past_key_values
        
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        response = self.forward_intercept(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position)

        if input_ids.size(1) > 1:
            del input_ids
            del attention_mask
            del position_ids
            del past_key_values
            del inputs_embeds
            del labels
            del use_cache
            del output_attentions
            del output_hidden_states
            del return_dict
            del cache_position
            gc.collect()
        
        return response

    def forward_intercept(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.context_position == 3:
            if input_ids.size(1) > 1:
                if len(cache_position.size()) == 1:
                    cache_position = cache_position.unsqueeze(0)
                self.pos_3_cache_position = torch.tensor([cache_position.size(1)], device=self.device).unsqueeze(0) 
                self.pos_3_attention_mask = attention_mask
            else:
                self.pos_3_cache_position += 1
                self.pos_3_position_ids += 1

                cache_position = self.pos_3_cache_position
                attention_mask = self.pos_3_attention_mask
                position_ids = self.pos_3_cache_position
                self.pos_3_idx += 1
            
        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        self.past_key_values = outputs.past_key_values

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # if self.context_position == 3 and input_ids.size(1) == 1:
        #     next_token_logits = logits[:, -1, :]
        #     next_token_id = torch.argmax(next_token_logits, dim=-1)
        #     #logger.info(f"\nnext_token_id (response): {next_token_id} {self.context_position} {len(input_ids)}\n")
        #     if 128009 in next_token_id:
        #         logger.info(f"\nnext_token_id (response): {next_token_id} {self.context_position} {len(input_ids)}\n")
        #         logger.info(f"\n\n==Ending response and saving interaction to memory buffer==\n\n")
        #         self.save_interaction()
        
        loss = None
        if labels is not None:

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
       
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):

        if past_key_values is not None:
            if inputs_embeds is not None:  
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]: 
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
   