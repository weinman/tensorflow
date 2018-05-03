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
#include <unordered_map>
#include <memory>
#include <vector>
#include <iostream>

#include "tensorflow/core/util/ctc/ctc_vocabulary.h"

namespace tensorflow {
namespace ctc {

class TrieNode {
  public:
    TrieNode(int vocabSize) : label(-1),
      prefixCount(0),
      vocabSize(vocabSize),
      endWord(false) {
        children.reserve(vocabSize);
      }

    TrieNode(int label, int vocabSize) : label(label),
      prefixCount(0),
      vocabSize(vocabSize),
      endWord(false) {
        children.reserve(vocabSize);
      }

    TrieNode(int label, int prefixCount, int vocabSize) : label(label),
      prefixCount(prefixCount),
      vocabSize(vocabSize),
      endWord(false) {
        children.reserve(vocabSize);
      }

    ~TrieNode() {
      children.clear();
    }

    // we're building the trie from a SparseTensorValue
    // each insertion is a dense vector of int labels
    void Insert(std::vector<int32> word) {
      // if word we are inserting is the end, then set the appropriate flag
      if (word.empty()) {
        endWord = true;
        return;
      }

      prefixCount++;
      int32 wordChar = word.at(0);
      if (wordChar <= vocabSize && wordChar >= 0) {
        // search for child node in word vector
        TrieNode *child_node = GetChildAt(wordChar);
        if (child_node == nullptr) {
          child_node = new TrieNode(wordChar, prefixCount, vocabSize);
          children.emplace(wordChar, child_node);
        }
        word.erase(word.begin());
        child_node->Insert(word);
      }
    }

    int32 GetLabel() {
      return label;
    }

    bool IsEnd() {
      return endWord;
    }

    TrieNode* GetChildAt(int32 label) {
      auto iter = children.find(label);
      if (iter != children.end())
        return iter->second;
      return nullptr;
    }

    std::vector<int32> GetTrieLabels() {
      std::vector<int32> labs;
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

    // DEPRECATED... alter function to use specified vocabulary size
    static void ReadFromStream(std::ifstream& in, TrieNode* &obj) {
      obj->ReadNode(in);

      for (int i = 0; i < obj->prefixCount; ++i) {
        TrieNode *c = new TrieNode(26);
        ReadFromStream(in, c);
        obj->children.insert({c->label, c});
      }
    }

  private:
    typedef std::unordered_map<int32, TrieNode*> TrieMap;

    int32 label;
    int prefixCount;
    int vocabSize;
    bool endWord;
    TrieMap children;

    void ReadNode(std::ifstream& in) {
      in >> label >> prefixCount;
    }

    std::vector<int32> __GetTrieLabels(std::vector<int32> labs) {
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
