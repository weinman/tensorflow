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

#ifndef TENSORFLOW_CURE_UTIL_CTC_CTC_VOCABULARY_H_
#define TENSORFLOW_CURE_UTIL_CTC_CTC_VOCABULARY_H_

#include <fstream>
#include <istream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tensorflow {
namespace ctc {

class Vocabulary {
  public:
    Vocabulary(const char *vocab_path) {
      // TODO: parse strings in path file to build up dictionary
      std::ifstream in(vocab_path, std::ios::in);
      ReadFromFile(in);
      in.close();
    }

    ~Vocabulary() {
      vocabulary.clear();
    }

    int GetVocabSize() {
      return vocab_size;
    }

    std::vector<int*> GetVocabList() {
      return vocabulary;
    }

  private:
    int vocab_size;
    std::vector<int*> vocabulary;

    void ReadFromFile(std::ifstream& in) {
      vocab_size = 0;
      std::string str;      
      while (std::getline(in, str)) {
        int ret[str.length()];
        for (int i=0; i<str.length(); ++i) {
          ret[i] = ((int) str.at(i)) - ((int) 'a');
        }
        vocabulary.push_back(ret);
        vocab_size++;        
      }
    }
};
 
} // namespace ctc
} // namespace tensorflow

#endif // CTC_VOCABULARY_H
