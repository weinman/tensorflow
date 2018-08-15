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

/* Code adapted from https://github.com/timediv/tensorflow-with-kenlm */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "ctc_trie_node.h"
#include "ctc_vocabulary.h"

using namespace tensorflow::ctc;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage " << argv[0]
              << " <vocabulary_path>"
              << " <trie_out_path>"
              << std::endl;
    return 1;
  }

  const char *vocabulary_path = argv[1];
  const char *trie_out_path = argv[2];

  Vocabulary vocabulary(vocabulary_path);
  std::vector<std::vector<char>> vocab_list = vocabulary.GetVocabList();

  // root of trie node has index -1
  // TODO: figure out more appropriate value
  TrieNode root(-1);
  for (std::vector<char> word : vocab_list) {
    root.Insert(word);
  }

  std::ofstream out;
  out.open(trie_out_path);
  root.WriteToStream(out);
  out.close();

  return 0;
}
