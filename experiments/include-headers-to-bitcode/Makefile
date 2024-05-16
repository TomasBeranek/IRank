# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the include-headers-to-bitcode experiment.
# ******************************************************************************

# This experiment shows:
# 	1) the bitcode generation property - header files are present in the
#			 generated bitcode of the source '.c' files
# 	2) llvm-slicer is able to slice according to the criteria present in the
#			 given header files

# Only point 2) alone accounts for 144,971 (~11%) of all samples - it needs to be handled.


# Example 1) can be run using 'make include-header'.

# The presence of code for functions whose definitions are in header.h can be
# verified by opening the graph in sliced.svg and checking that the constant
# '12345' is created in the graph.


# Example 2) can be run using 'make slice-by-header'.

# Slicing by code in header cannot be done in the same way as for '.c' files - the
# bitcode probably does not contain the information that the code comes from
# header.h or llvm-slicer cannot find it. However, it is possible to slice using
# a modified criterion that provides the same functionality - slicing by function
# name and line in the original header.h (not by line in the function). The
# result of the experiment can again be verified in the sliced.svg graph by
# finding the constant '12345'.


# The following are examples of real samples representing cases 1) and 2).


# Case 1) - we slice by '.c', but we need code from '.h'.

# Bug info
# 	Line from slicing criteria extraction:
#			0,openssl_289c6a8719428a1a555c0aa64d30e54b4e7c6b97_1,get_dsa512,bn_lib.c,459,
#		Commit: 4e8172d6dafec18c4512267b80879de7f0a9fff9
#		Files: 'apps/testdsa.h', 'crypto/bn/bn_lib.c'

#   'qualifier': 'Offset: [-oo, +oo] (⇐ [-oo, +oo] + [-oo, +oo]) Size: [0, +oo] by call to `BN_bin2bn`.',
#   'file': 'apps/testdsa.h',
# 	'procedure': 'get_dsa512',
# 	'line': 44,
# 	'column': 16,
# 	'url': 'https://github.com/openssl/openssl/blob/4e8172d6dafec18c4512267b80879de7f0a9fff9/apps/testdsa.h/#L44'}
# 	"trace": [
# 			{
# 					"level": 0,
# 					"description": "Call",
# 					"file": "apps/testdsa.h",
# 					"loc": "44:16",
# 					"func_name": "get_dsa512",
# 			},
# 			{
# 					"level": 1,
# 					"description": "Parameter `ret->top`",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "664:1",
# 					"func_name": "BN_bin2bn",
# 			},
# 			{
# 					"level": 1,
# 					"description": "Call",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "680:6",
# 					"func_name": "BN_bin2bn",
# 			},
# 			{
# 					"level": 0,
# 					"description": "<Offset trace>",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "440:1",
# 					"func_name": "bn_expand2",
# 			},
# 			{
# 					"level": 2,
# 					"description": "Parameter `words`",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "440:1",
# 					"func_name": "bn_expand2",
# 			},
# 			{
# 					"level": 0,
# 					"description": "<Length trace>",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "440:1",
# 					"func_name": "bn_expand2",
# 			},
# 			{
# 					"level": 2,
# 					"description": "Parameter `*b->d`",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "440:1",
# 					"func_name": "bn_expand2",
# 			},
# 			{
# 					"level": 2,
# 					"description": "Array access: Offset: [-oo, +oo] (\u21d0 [-oo, +oo] + [-oo, +oo]) Size: [0, +oo] by call to `BN_bin2bn` ",
# 					"file": "crypto/bn/bn_lib.c",
# 					"loc": "459:3",
# 					"func_name": "bn_expand2",
# 			}]


# Case 2) - we slice by '.h'.

# Bug info
# 	Line from slicing criteria extraction:
#			0,ffmpeg_4ac530632ac022552b970d1f0d52e3c01ba74edc_0,vc1_decode_b_blocks,get_bits.h,851,
#		Commit: 6847affcb775e89d9256b33c7dc8a7502d2bb428
#		Files: 'libavcodec/vc1_block.c', 'libavcodec/get_bits.h'

#		"qualifier": "(`v->s.gb.size_in_bits` - [-oo, +oo]):signed32 by call to `get_bits_left`.",
# 	"file": "libavcodec/vc1_block.c",
# 	"procedure": "vc1_decode_b_blocks",
# 	"line": 2954,
# 	"column": 17,
# 	"adjusted_bug_loc": {
#     "file": "libavcodec/get_bits.h",
#     "line": 851,
#     "column": 5,
# 	},
# 	"trace": [
#     	{
#         	"level": 0,
#         	"description": "Parameter `v->s.gb.size_in_bits`",
#         	"file": "libavcodec/vc1_block.c",
#         	"loc": "2899:1",
#         	"func_name": "vc1_decode_b_blocks",
#     	},
#     	{
#         	"level": 0,
#         	"description": "Call",
#         	"file": "libavcodec/vc1_block.c",
#         	"loc": "2954:17",
#         	"func_name": "vc1_decode_b_blocks",
#     	},
#     	{
#         	"level": 0,
#         	"description": "<LHS trace>",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "849:1",
#         	"func_name": "get_bits_left",
#     	},
#     	{
#         	"level": 1,
#         	"description": "Parameter `gb->size_in_bits`",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "849:1",
#         	"func_name": "get_bits_left",
#     	},
#     	{
#         	"level": 0,
#         	"description": "<RHS trace>",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "849:1",
#         	"func_name": "get_bits_left",
#     	},
#     	{
#         	"level": 1,
#         	"description": "Parameter `gb->index`",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "849:1",
#         	"func_name": "get_bits_left",
#     	},
#     	{
#         	"level": 1,
#         	"description": "Call",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "851:31",
#         	"func_name": "get_bits_left",
#     	},
#     	{
#         	"level": 2,
#         	"description": "Parameter `s->index`",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "219:1",
#         	"func_name": "get_bits_count",
#     	},
#     	{
#         	"level": 2,
#         	"description": "Assignment",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "224:5",
#         	"func_name": "get_bits_count",
#     	},
#     	{
#         	"level": 1,
#         	"description": "Binary operation: (v->s.gb.size_in_bits - [-oo, +oo]):signed32 by call to `get_bits_left` ",
#         	"file": "libavcodec/get_bits.h",
#         	"loc": "851:5",
#         	"func_name": "get_bits_left",
#     	}]


# This wild syntax is needed in order to include `#` in the slicing criteria variable
define ONLY_INCLUDE_HEADER_SC
--entry=fun2 --sc "file.c##6#"
endef
export ONLY_INCLUDE_HEADER_SC

define SLICE_BY_HEADER_SC
--entry=fun2 --sc "#fun2#8#"
endef
export SLICE_BY_HEADER_SC

all: include-header

clean:
	rm -rf a.out infer-out/ sliced.dot sliced.svg header.h.gch sliced.bc file.bc

infer: clean
	infer --bufferoverrun -- clang file.c

file.bc:
	# Create bitcode
	clang -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -c file.c header.h

include-header: clean file.bc
	# Slice
	llvm-slicer $(ONLY_INCLUDE_HEADER_SC) -o=sliced.bc file.bc
	# Dump SVG graph
	llvm-slicer --dump-dg-only $(ONLY_INCLUDE_HEADER_SC) sliced.bc
	dot -Tsvg sliced.dot > sliced.svg

slice-by-header: clean file.bc
	# Slice
	llvm-slicer $(SLICE_BY_HEADER_SC) -o=sliced.bc file.bc
	# Dump SVG graph
	llvm-slicer --dump-dg-only $(SLICE_BY_HEADER_SC) sliced.bc
	dot -Tsvg sliced.dot > sliced.svg
