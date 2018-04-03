/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/ctc_decoder_ops.cc"

namespace tensorflow {

// CTC beam search trie
class CTCBeamSearchDecoderTrieOp : public OpKernel {
 public:
  explicit CTCBeamSearchDecoderTrieOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
    int top_paths;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
    decode_helper_.SetTopPaths(top_paths);
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* inputs;
    const Tensor* seq_len;
    const Tensor* dictionary = nullptr;
    Tensor* log_prob = nullptr;
    OpOutputList decoded_indices;
    OpOutputList decoded_values;
    OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    auto inputs_t = inputs->tensor<float, 3>();
    auto seq_len_t = seq_len->vec<int32>();
    // TODO: get dictionary length
    auto dictionary_t = dictionary->tensor<int32, 1>();
    auto log_prob_t = log_prob->matrix<float>();

    const TensorShape& inputs_shape = inputs->shape();

    const int64 max_time = inputs_shape.dim_size(0);
    const int64 batch_size = inputs_shape.dim_size(1);
    const int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    log_prob_t.setZero();

    std::vector<TTypes<float>::UnalignedConstMatrix> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
    }

    std::vector<std::vector<char>> dictionary_vec;
    ctc::TrieBeamScorer beam_scorer_(dictionary_vec, true);
    ctc::CTCBeamSearchDecoder<ctc::TrieBeamState> beam_search(num_classes,
                                                         beam_width_,
                                                         &beam_scorer_,
                                                         1 /* batch_size */,
                                                         merge_repeated_);
    Tensor input_chip(DT_FLOAT, TensorShape({num_classes}));
    auto input_chip_t = input_chip.flat<float>();

    std::vector<std::vector<std::vector<int> > > best_paths(batch_size);
    std::vector<float> log_probs;

    // Assumption: the blank index is num_classes - 1
    for (int b = 0; b < batch_size; ++b) {
      auto& best_paths_b = best_paths[b];
      best_paths_b.resize(decode_helper_.GetTopPaths());
      for (int t = 0; t < seq_len_t(b); ++t) {
        input_chip_t = input_list_t[t].chip(b, 0);
        auto input_bi =
            Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
        beam_search.Step(input_bi);
      }
      OP_REQUIRES_OK(
          ctx, beam_search.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b,
                                    &log_probs, merge_repeated_));

      beam_search.Reset();

      for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
        log_prob_t(b, bp) = log_probs[bp];
      }
    }

    OP_REQUIRES_OK(ctx, decode_helper_.StoreAllDecodedSequences(
                            best_paths, &decoded_indices, &decoded_values,
                            &decoded_shape));
  }

 private:
  CTCDecodeHelper decode_helper_;
  bool merge_repeated_;
  int beam_width_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderTrieOp);
};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchDecoderTrie").Device(DEVICE_CPU),
                        CTCBeamSearchDecoderTrieOp);

}  // end namespace tensorflow
