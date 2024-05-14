# This experiment shows, that llvm-slicer is capable of slicing by given file
# and line.

# To run the experiment, run 'make dot', which creates two .svg LLVM graphs
# -- all.svg (unsliced) and sliced.svg (sliced). In the experiment, we are slicing
# according to line 24, so everything but part 1 and line 24 should be removed.
# This can be checked by looking in all.svg/line_slicing_criteria.dot for two
# multiplications ('mul' instructions) representing operations on the
# lines 16 and 22. When the program is sliced, these lines should be removed,
# which can be confirmed by checking that there is no mul instruction in
# sliced.svg/out.dot.

# This example also shows that llvmslicer preserves everything on the line it
# is slicing against. Again, this can be checked by seeing that the 'sub'
# instruction is present in sliced.svg/out.dot, corresponding to the subtraction
# on line 24.

# The problem arises if chunks of code are uncommented, then llvm-slicer is
# unable to determine what variable the 'q' pointer refers to,

# View .dot graphs e.g. here: https://dreampuf.github.io/GraphvizOnline/ or use
# created .svg files

all: slice

line_slicing_criteria.bc:
	clang -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -c line_slicing_criteria.c

clean:
	rm -f line_slicing_criteria.bc out.bc all.svg sliced.svg line_slicing_criteria.dot out.dot line_slicing_criteria.sliced

compile: clean line_slicing_criteria.bc

# Should slice away everything except part 1 and line 24
slice: compile
	llvm-slicer --entry=main --sc "line_slicing_criteria.c##24#" -o=out.bc line_slicing_criteria.bc

# Doesn't slice but still need slicing criteria (just a bug?)
dot-all:
	llvm-slicer --dump-dg-only --sc="line_slicing_criteria.c##24#" line_slicing_criteria.bc
	dot -Tsvg line_slicing_criteria.dot > all.svg
	# llvm-slicer --annotate=slice --dump-dg-only --sc="line_slicing_criteria.c##24#" line_slicing_criteria.bc

dot-sliced:
	llvm-slicer --dump-dg-only --sc="line_slicing_criteria.c##24#" out.bc
	dot -Tsvg out.dot > sliced.svg
	# llvm-slicer --annotate=slice --dump-dg-only --sc="line_slicing_criteria.c##24#" out.bc

dot: slice dot-all dot-sliced

# llvm-dg-dump is deprecated and doesn't support this type of slicing criteria
deprecated-dot: compile
	llvm-dg-dump --mark --sc="line_slicing_criteria.c##24#" line_slicing_criteria.bc > graph.dot
