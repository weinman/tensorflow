# Description:
#   Contains op to run ctc with a trie-implemented dictionary-restricted framework

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_gen_op_wrapper_py",
)

tf_custom_op_py_library(
    name = "ctc_beam_search_decoder_trie_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [
        ":python/ops/_ctc_beam_search_decoder_trie_ops.so",
    ],
    kernels = [
        ":ctc_beam_search_decoder_trie_ops_kernels",
        ":ctc_beam_search_decoder_trie_ops_op_lib",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":ctc_beam_search_decoder_trie_ops",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python:ctc_ops",
    ],
)

tf_custom_op_library(
    name = "python/ops/_ctc_beam_search_decoder_trie_ops.so",
    srcs = [
        "kernels/ctc_beam_search_decoder_trie_ops.cc",
        "ops/ctc_beam_search_decoder_trie_ops.cc"
    ],
    deps = [
        "//tensorflow/core/util/ctc:ctc_beam_search_lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core:ctc_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "ctc_beam_search_decoder_trie_ops",
    deps = [":ctc_beam_search_decoder_trie_ops_op_lib"],
)

tf_gen_op_libs(
    op_lib_names = ["ctc_beam_search_decoder_trie_ops"],
)

tf_kernel_library(
    name = "ctc_beam_search_decoder_trie_ops_kernels",
    prefix = "kernels/ctc_beam_search_decoder_trie_ops",
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:eigen_helpers",
        "//third_party/eigen3",
    ],
)
