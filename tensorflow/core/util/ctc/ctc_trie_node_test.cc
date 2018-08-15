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

#include <iostream>
#include <string>
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ctc/ctc_trie_node.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"

namespace {

using tensorflow::ctc::TrieNode;
using tensorflow::ctc::Vocabulary;

const int32 test_labels_count = 35;
const int32 test_labels[] = {-1, 19, 7, 4, 16, 20, 8, 2, 10, 1, 17, 14, 22, 13,
                            5, 14, 23, 9, 20, 12, 15, 4, 3, 14, 21, 4, 17, 0,
                            11, 0, 25, 24, 3, 14, 6};

const char *vocabulary_path = "./tensorflow/core/util/ctc/testdata/vocab";

void __printTrie(TrieNode *root) {
  std::cout << root->GetLabel() << std::endl;
  for(TrieNode * c : root->GetChildren()) {
    __printTrie(c);
  }
}

TEST(TrieNode, VocabularyFromFile) {
  Vocabulary vocabulary(vocabulary_path);
  EXPECT_EQ(9, vocabulary.GetVocabSize());
}

TEST(TrieNode, TrieConstructionTest) {
  Vocabulary vocabulary(vocabulary_path);

  TrieNode root(-1);
  std::vector<std::vector<int32>> vocab_list = vocabulary.GetVocabList();
  for (std::vector<int32> word : vocab_list) {
    root.Insert(word);
  }

  __printTrie(&root);

  std::vector<int32> node_labs = root.GetTrieLabels();
  std::cout << "Vector size:  " << node_labs.size() << std::endl;
  for (int i=0; i < node_labs.size(); ++i) {
    EXPECT_EQ(node_labs.at(i), test_labels[i]);
  }
}

} // namespace
