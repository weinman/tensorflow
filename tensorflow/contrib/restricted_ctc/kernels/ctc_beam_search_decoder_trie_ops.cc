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

#define EIGEN_USE_THREADS

#include <limits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/ctc/ctc_beam_search.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

class CTCDecodeHelper {
 public:
  CTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  Status ValidateInputsGenerateOutputs(
      OpKernelContext* ctx, const Tensor** inputs, const Tensor** seq_len,
      Tensor** log_prob, OpOutputList* decoded_indices,
      OpOutputList* decoded_values, OpOutputList* decoded_shape) const {
    Status status = ctx->input("inputs", inputs);
    if (!status.ok()) return status;
    status = ctx->input("sequence_length", seq_len);
    if (!status.ok()) return status;

    const TensorShape& inputs_shape = (*inputs)->shape();

    if (inputs_shape.dims() != 3) {
      return errors::InvalidArgument("inputs is not a 3-Tensor");
    }

    const int64 max_time = inputs_shape.dim_size(0);
    const int64 batch_size = inputs_shape.dim_size(1);

    if (max_time == 0) {
      return errors::InvalidArgument("max_time is 0");
    }
    if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
      return errors::InvalidArgument("sequence_length is not a vector");
    }

    if (!(batch_size == (*seq_len)->dim_size(0))) {
      return errors::FailedPrecondition(
          "len(sequence_length) != batch_size.  ",
          "len(sequence_length):  ", (*seq_len)->dim_size(0),
          " batch_size: ", batch_size);
    }

    auto seq_len_t = (*seq_len)->vec<int32>();

    for (int b = 0; b < batch_size; ++b) {
      if (!(seq_len_t(b) <= max_time)) {
        return errors::FailedPrecondition("sequence_length(", b,
                                          ") <= ", max_time);
      }
    }

    Status s = ctx->allocate_output(
        "log_probability", TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;

    s = ctx->output_list("decoded_indices", decoded_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", decoded_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", decoded_shape);
    if (!s.ok()) return s;

    return Status::OK();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int> > >& sequences,
      OpOutputList* decoded_indices, OpOutputList* decoded_values,
      OpOutputList* decoded_shape) const {
    // Calculate the total number of entries for each path
    const int64 batch_size = sequences.size();
    std::vector<int64> num_entries(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto& batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      Tensor* p_indices = nullptr;
      Tensor* p_values = nullptr;
      Tensor* p_shape = nullptr;

      const int64 p_num = num_entries[p];

      Status s =
          decoded_indices->allocate(p, TensorShape({p_num, 2}), &p_indices);
      if (!s.ok()) return s;
      s = decoded_values->allocate(p, TensorShape({p_num}), &p_values);
      if (!s.ok()) return s;
      s = decoded_shape->allocate(p, TensorShape({2}), &p_shape);
      if (!s.ok()) return s;

      auto indices_t = p_indices->matrix<int64>();
      auto values_t = p_values->vec<int64>();
      auto shape_t = p_shape->vec<int64>();

      int64 max_decoded = 0;
      int64 offset = 0;

      for (int64 b = 0; b < batch_size; ++b) {
        auto& p_batch = sequences[b][p];
        int64 num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        std::copy_n(p_batch.begin(), num_decoded, &values_t(offset));
        for (int64 t = 0; t < num_decoded; ++t, ++offset) {
          indices_t(offset, 0) = b;
          indices_t(offset, 1) = t;
        }
      }

      shape_t(0) = batch_size;
      shape_t(1) = max_decoded;
    }
    return Status::OK();
  }

 private:
  int top_paths_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCDecodeHelper);
};

// CTC beam search trie
class CTCBeamSearchDecoderTrieOp : public OpKernel {
 public:
  explicit CTCBeamSearchDecoderTrieOp(OpKernelConstruction* ctx):
   OpKernel(ctx) {
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
