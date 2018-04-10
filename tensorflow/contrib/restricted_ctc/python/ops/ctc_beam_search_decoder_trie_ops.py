# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor

from tensorflow.python.ops import gen_ctc_ops
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_ctc_beam_search_decoder_trie_so = loader.load_op_library(
    resource_loader.get_path_to_datafile(
        "_ctc_beam_search_decoder_trie_ops.so"))

def ctc_beam_search_decoder_trie(inputs, sequence_length, dictionary,
                                 beam_width=100, top_paths=1,
                                 merge_repeated=True):
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      gen_ctc_ops.ctc_beam_search_decoder_trie(
          inputs, sequence_length, dictionary, beam_width=beam_width,
          top_paths=top_paths, merge_repeated=merge_repeated))

  return (
      [sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)

ops.NotDifferentiable("CTCBeamSearchDecoderTrie")
