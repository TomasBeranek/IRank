# Master's Thesis

### Evaluating Reliability of Static Analysis Results Using Machine Learning

## Thesis Details

##### Author:  [Bc. Tomáš Beránek](https://www.vut.cz/en/people/tomas-beranek-213297)
##### Language:  EN
##### Category:  Artificial Intelligence
##### Company:  Red Hat Czech s.r.o.


##### Supervisor:  [prof. Ing. Tomáš Vojnar, Ph.D.](https://www.vut.cz/lide/tomas-vojnar-2491)
##### Consultants:  [Mgr. Marek Grác, Ph.D.](https://www.vut.cz/lide/marek-grac-199772), [Ing. Viktor Malík](https://www.vut.cz/lide/viktor-malik-143967)

### Specification

  1. Learn about the Infer tool for static analysis and finding bugs in software.
  2. Explore the possibilities of using machine learning in the context of source code analysis.
  3. Get a dataset containing the bugs found by Infer, with information on whether they are real bugs or not.
  4. Design and implement a machine learning based approach (using the dataset obtained in Section 3) that will be able to automatically determine whether a bug found by Infer represents a real bug or not.
  5. Evaluate the quality of your solution on at least two different open-source projects.
  6. Summarize and discuss the results obtained and their possible extensions.

### Literature
 - [Facebook Infer](https://fbinfer.com/)
 - Cao, Sicong, et al. "Bgnn4vd: constructing bidirectional graph neural-network for vulnerability detection." Information and Software Technology 136 (2021): 106576.
 - Y. Zheng et al., "D2A: A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis," 2021 IEEE/ACM 43rd International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP), 2021, pp. 111-120.

## Notes

#### Extrakce grafu -- Joern
  - běží až na LLVM IR (po extrakci pomocí [llvm2cpg](https://github.com/ShiftLeftSecurity/llvm2cpg)) -> makra jsou už rozbalená
  - také se jedná o udržovanější repo (proto vybírám tento přístup)
  - [podporované jazyky](https://docs.joern.io/home#supported-languages)
  - [dokumentace](https://docs.joern.io/home)
  - instalace Joern:
    1. ```cd code-extraction/```
    2. ```git clone https://github.com/joernio/joern```
    3. ```cd joern/```
    4. ```sudo ./joern-install.sh```
    5. ```joern```
  - instalace [llvm2cpg](https://github.com/ShiftLeftSecurity/llvm2cpg/releases)
    1. stáhnout binary release
    2. ```unzip unzip llvm2cpg-0.8.0-LLVM-11.0-ubuntu-20.04.zip```
    3. ```mv llvm2cpg-0.8.0-LLVM-11.0-ubuntu-20.04/ llvm2cpg```
    4. ```sudo ln -s $PWD/llvm2cpg/llvm2cpg /usr/bin/llvm2cpg```


#### Program slicing -- pomocí DG
  - DG je [nástroj](https://github.com/mchalupa/dg) pro analýzu kódu a program slicing
  - je k němu i [paper](https://www.fi.muni.cz/~xchalup4/dg_atva20_preprint.pdf)
  - návod na program slicing [zde](https://github.com/mchalupa/dg/blob/master/doc/llvm-slicer.md)
  - nefunguje dobře na C++ podle [toho](https://github.com/mchalupa/dg/blob/master/doc/llvm-slicer.md#using-slicer-on-c-bitcode)
  - instalace DG pomocí bínárky pro Ubuntu 18.04 je zastaralé -- více jak 2 roky
  - je k dispozici i [návod](https://github.com/mchalupa/dg/blob/master/doc/compiling.md) na překlad:
    1. ```sudo apt install git cmake make llvm zlib1g-dev clang g++ python3```
    2. ```cd code-extraction/```
    3. ```git clone https://github.com/mchalupa/dg```
    4. ```cd dg```
    5. ```mkdir build```
    6. ```cd build/```
    7. ```cmake ..```
    8. ```make -j4```
    9. ```make check``` -- optional pro spuštění testů
    10. ```sudo ln -s /home/tomas/Documents/diplomka/code-extraction/dg/build/tools/llvm-slicer /usr/bin/llvm-slicer```
  - program slicing (už je nutno mít vytvořené ```.bc``` soubory)
    1. ```llvm-link add_one.bc main.bc -o bitcode.bc```
    2. ```llvm-slicer -c 21:x bitcode.bc``` -- slicing podle proměnné ```x``` na řádku ```21```, vytvoří se soubor ```bitcode.sliced```
    3. ```llvm2cpg bitcode.sliced --output=./main.cpg.bin.zip```
  - pokud chci specifikovat i soubor, kde byla proměnná použita lze to pomocí ```-sc add_one.c##7#&p```, je to zmíněno [zde](https://github.com/mchalupa/dg/issues/350)
