# ******************************* Makefile **********************************
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author: Beranek Tomas (xberan46)
#  Date: 31.12.2022
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
# ***************************************************************************

# TODO https://www.gnu.org/software/make/manual/html_node/Standard-Targets.html

COMPILERS = 8cc 9cc ack c++ ccomp chibicc clang cproc g++ gcc icc icpc lacc lcc openCC opencc pcc scc sdcc tcc vc x86_64-redhat-linux-c++ x86_64-redhat-linux-g++ x86_64-redhat-linux-gcc x86_64-redhat-linux-gcc-10
COMPILERS_PATHS = $(addprefix /usr/bin/, $(COMPILERS))
ORIGINAL_COMPILERS_PATHS = $(addsuffix -original, $(COMPILERS_PATHS))

# Since $(COMPILERS_PATHS) and $(ORIGINAL_COMPILERS_PATHS) are absolute paths, the targets with the same name won't be executed if the given files exist.
.PHONY : all install-dependencies install-wrappers uninstall-wrappers install-pipeline $(COMPILERS_PATHS) $(ORIGINAL_COMPILERS_PATHS)

all: install-dependencies install-wrapper

install-dependencies:
	@>&2 echo 'TBD'

install-wrappers: $(COMPILERS_PATHS)

$(COMPILERS_PATHS):
	@if [ -e $@ ]; then \
		if [ ! -e $@-original ]; then \
			sudo mv $@ $@-original; \
			sudo cp ./wrapper $@; \
			sudo sed -i 's/COMPILER_NAME/$(subst /usr/bin/,,$@)/g' $@; \
			>&2 echo 'NOTE: $@ wrapper has been successfully installed!'; \
		else \
			sudo cp ./wrapper $@; \
			sudo sed -i 's/COMPILER_NAME/$(subst /usr/bin/,,$@)/g' $@; \
			>&2 echo 'NOTE: $@ wrapper has been successfully reinstalled!'; \
		fi; \
	fi

uninstall-wrappers: $(ORIGINAL_COMPILERS_PATHS)

$(ORIGINAL_COMPILERS_PATHS):
	@if [ -e $@ ]; then \
		sudo mv $@ $(@:-original=); \
		>&2 echo 'NOTE: $(@:-original=) wrapper has been successfully uninstalled!'; \
	fi

install-pipeline: install-wrappers
