#!/bin/bash

python -m src.tasks.GenVQA --encoder_type lxmert --decoder_type transformer --rnn_type lstm --num_rnn_layers 1 --bidirectional --attn_type bahdanau --attn_method dot --nheads 12 --num_transformer_layer 3