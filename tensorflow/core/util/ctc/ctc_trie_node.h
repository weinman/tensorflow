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

#ifndef TENSORFLOW_CORE_UTIL_CTC_CTC_TRIE_NODE_H_
#define TENSORFLOW_CORE_UTIL_CTC_CTC_TRIE_NODE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>

// #include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/util/ctc/ctc_vocabulary.h"

namespace tensorflow {
namespace ctc {

class TrieNode {
  public:
    TrieNode() : label(-1),
      prefixCount(0) {
        children = TrieMap(26);
    }

    TrieNode(int label) : label(label),
      prefixCount(0) { }

    ~TrieNode() {
      children.clear();
    }

    // we're building the trie from a SparseTensorValue
    // each insertion is a dense vector of int labels
    void Insert(std::vector<char> word) {
      if (word.empty()) return;
      prefixCount++;
      int wordChar = word.at(0);
      if (wordChar <= 26 && wordChar >=0 ) {
        // search for child node in word vector
        TrieNode *child = GetChildAt(wordChar);
        if (child == nullptr) {
          children.emplace(wordChar, child);
        }
        word.erase(word.begin());
        child->Insert(word);
      }
    }

    char GetLabel() {
      return label;
    }

    TrieNode* GetChildAt(char label) {
      auto iter = children.find(label);
      if (iter == children.end()) {
        return nullptr;
      } else {
        return iter->second;
      }
    }

    std::vector<char> GetTrieLabels() {
      std::vector<char> labs;
      labs = __GetTrieLabels(labs);
      return labs;
    }

    std::vector<TrieNode*> GetChildren() {
      std::vector<TrieNode*> nodes;
      for (const auto& c : children) {
        nodes.push_back(c.second);
      }
      return nodes;
    }

    void WriteToStream(std::ofstream& out) {
      out << label << " " << prefixCount << std::endl;
      // recursive call
      for (const auto& c : children) {
        c.second->WriteToStream(out);
      }
    }

    static void ReadFromStream(std::ifstream& in, TrieNode* &obj) {
      obj->ReadNode(in);

      for (int i = 0; i < obj->prefixCount; ++i) {
        TrieNode *c = new TrieNode();
        ReadFromStream(in, c);
        obj->children.emplace(c->label, c);
      }
    }

  private:
    typedef gtl::FlatMap<char, TrieNode*> TrieMap;

    char label;
    int prefixCount;
    TrieMap children;

    void ReadNode(std::ifstream& in) {
      in >> label >> prefixCount;
    }

    std::vector<char> __GetTrieLabels(std::vector<char> labs) {
      labs.push_back(label);
      for (const auto& c : children) {
        labs = c.second->__GetTrieLabels(labs);
      }
      return labs;
    }

}; // TrieNode
} // namespace ctc
} // namespace tensorflow

#endif
