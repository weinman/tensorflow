# Description:
#   Contains op to run ctc with a trie-implemented dictionary-restricted framework

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_cc_test",
    "tf_py_test",
    "tf_gen_op_libs",
    "tf_kernel_library",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_kernel_tests_linkstatic",
)

cc_library(
    name = "all_ops",
    deps = ["ctc_beam_search_decoder_trie_ops_op_lib"],
)

cc_library(
    name = "all_kernels",
    deps = ["ctc_beam_search_decoder_trie_ops_kernels"],
)

tf_custom_op_py_library(
    name = "ctc_beam_search_decoder_trie_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    dso = [
        ":python/ops/_ctc_beam_search_decoder_ops.so",
    ],
    kernels = [
        ":all_ops",
        ":all_kernels",
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
    name = "python/ops/_ctc_beam_search_decoder_trie.so",
    srcs = [
        "kernels/ctc_beam_search_decoder_trie_ops.cc",
        "ops/ctc_beam_search_decoder_trie.cc"
    ],
    deps = [
        "//tensorflow/core/kernels:eigen_helpers",
        "//tensorflow/core/kernels:ctc_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "ctc_beam_search_decoder_trie_ops",
    deps = [":ctc_beam_search_decoder_trie_ops_op_lib"],
)

tf_gen_op_libs(
    op_lib_names = [
        "ctc_beam_search_decoder_trie_ops",
    ],
)

tf_kernel_library(
    name = "ctc_beam_search_decoder_trie_ops_kernels",
    prefix = "kernels/ctc_beam_search_decoder_trie_ops",
    deps = [
        "//tensorflow/core:ctc_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/util/ctc:ctc_beam_search_lib",
        "//tensorflow/core/util/ctc:ctc_loss_calculator_lib",
    ],
)
