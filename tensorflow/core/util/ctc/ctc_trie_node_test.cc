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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ctc/ctc_trie_node.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"

namespace {

using tensorflow::ctc::TrieNode;
using tensorflow::ctc::Vocabulary;

const char test_sentence[] = "the quick brown fox jumps over the lazy dog";
const char test_labels_count = 34;
const char test_labels[] = {19, 7, 4, 16, 20, 8, 2, 10, 1, 17, 14, 22, 13, 5, 14,
                            23, 9, 20, 12, 15, 4, 3, 14, 21, 4, 17, 0, 11, 0, 25, 24,
                            3, 14, 6};

const char *test_directory_path = "./tensorflow/core/util/ctc/testdata"
const char *vocabulary_path = "./tensorflow/core/util/ctc/testdata/vocab"

TrieNode *createTrieNode() {
  TrieNode *node;
  ifstream in;
  in.open(vocabulary_path);
  ReadFromStream(in, node);
  in.close();
  return node;
}

TEST(TrieNode, Vocabulary) {
  TrieNode *node = createTrieNode();
  std::vector<int> node_labs = node->GetTrieLabels();
  for (int i=0; i < node_labs.size(); ++i) {
    EXPECT_EQ(node_labs.at(i), test_labels[i]);
  }
}
  
} // namespace
