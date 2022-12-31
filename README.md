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

#### Reprezentace kódu
  - BGNN4VD používá __CCG__ (Code Composite Graph) = AST, CFG, DFG
  - zkusím použít __CPG__ (Code Property Graph) = AST, CFG, PDG (Program Dependence Graph)
    - PDG by mělo obsahovat to stejné jako DFG + pár věcí navíc (jak nějaké predikáty ovlivňují
    různé kusy kódu) -> model bude mít __více užitečných informací__
    - [blog](https://blog.embold.io/code-representation-for-machine-learning-code-as-graph/) o CPG
    - [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6956589) o CPG
    - [repo](https://github.com/joernio/joern) __[nové]__ pro extrakci CPG zejména z LLVM IR (ale i z jiných zdrojů)
    - [repo](https://github.com/Fraunhofer-AISEC/cpg) __[staré]__ pro extrakci CPG z C/C++/Java (experimentálně pro Python, Golang, TypeScript)
      - má podporu i pro [LLVM IR](https://llvm.org/docs/LangRef.html) (zřejmě půjde napojit na Infer clang)

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
  - instalace [neo4j](https://linuxhint.com/install-neo4j-ubuntu/), něco převzato i z [návodu](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04)
    1. ```sudo apt update```
    2. ```sudo apt install apt-transport-https ca-certificates curl software-properties-common```
    3. ```sudo curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -``` -- nepoužívat
    4. ```sudo add-apt-repository “deb https://debian.neo4j.com stable 4.1”``` -- nepoužívat
    5. ```curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key |sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg```
    6. ```echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 4.1" | sudo tee -a /etc/apt/sources.list.d/neo4j.list```
    7. ```sudo apt update```
    8. ```sudo apt install neo4j```
    9. ```sudo systemctl status neo4j.service``` -- ověření statusu
    10. ```sudo systemctl stop neo4j.service``` -- pro zastavení
    11. ```sudo systemctl start neo4j.service``` -- pro start
    12. v prohlížeči na adrese [localhost:7474](localhost:7474) se přihlásit:
     - username: ```neo4j```
     - password: ```neo4j```
    13. změnit heslo např. ```123```
  - extrakce LLVM IR (při překladu) a vygenerování CPG
    1. ```clang -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -c main.c```
    2. ``` llvm2cpg `find ./ -name "*.bc"` --output=./main.cpg.bin.zip ```
  - načtení CPG do Joern a následně do neo4j
    1. ```joern```
    2. ```workspace.reset```
    3. ```importCpg("./main.cpg.bin.zip")```
    4. ```save```
    5. ```exit```
    6. ```joern-export --repr all --format neo4jcsv workspace/main.cpg.bin.zip/cpg.bin```
    7. ```sudo cp /home/tomas/Documents/diplomka/code-extraction/example/out/*_data.csv /var/lib/neo4j/import/```
    8. ```find /home/tomas/Documents/diplomka/code-extraction/example/out -name 'nodes_*_cypher.csv' -exec /bin/cypher-shell -u neo4j -p 123 --file {} \;```
    9. ```find /home/tomas/Documents/diplomka/code-extraction/example/out -name 'edges_*_cypher.csv' -exec /bin/cypher-shell -u neo4j -p 123 --file {} \;```
    10. ```MATCH (n) RETURN n``` -- zobrazení v prohlížeči
    11. ```DETACH DELETE``` -- smazaní celé DB
    12. ```sudo rm /var/lib/neo4j/import/*``` -- vyčištění importu

  - pokud CPG jednotlivých souborů nepůjde spojit, tak by možná šlo využít [WLLVM](https://github.com/travitch/whole-program-llvm)

#### Extrakce grafu -- staré
  - nebere v potaz překladové příkazy -> tudíž nerozbaluje makra
  - build CPG knihovny jako CLI toolu (cca 3m):
   1. ```cd code-extraction/```
   2. ```git clone https://github.com/Fraunhofer-AISEC/cpg.git```
   3. ```cd cpg/cpg-console/```
   4. ```../gradlew installDist```
  - [instalace Neo4j](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-neo4j-on-ubuntu-20-04)
  - vygenerování grafu z ```code-extraction/example/main.c```:
   1. ```./build/install/cpg-console/bin/cpg-console```
   2. ```:tr ../../example/main.c``` (při úspěchu by měl program vypsat ```03:08:33,948 INFO  MeasurementHolder TranslationManager: Translation into full graph done in 366 ms```)
   3. ```:export neo4j```


#### Verze extrakce grafů
  1. CPG knihovna
    - není dokumentována extrace cpg z LLVM IR/BC
    - analýza tudíž pouze na zdrojových kódech
  2. Joern
    - v základu opět pouze na zdrojových kódech
  3. Joern + llvm2cpg + -emit-llvm
    - Joern lze propojit s nástrojem llvm2cpg a načíst tak graf z LLVM IR/BC
    - -emit-llvm lze spustit pouze při překladu souborů na object soubory
    - tento přístup dokáže rozbalit makra
    - běží nad LLVM IR/BC
    - 🔴__možná by šlo__🔴 použít ```llvm2cpg `find ./ -name "*.ll"``` a třeba to ty soubory samo spojí samo a nebylo nutné hledat binárky/knihovny -- 🔴nutné otestovat🔴
  3. Joern + llvm2cpg + -fembed-bitcode
    - LLVM IR/BC je uloženo ve výstupní binárce/knihovně (nutno otestovat s knihovnou) a vystupná graf by měl být zřejmě jeden (nutno otestovat)
  4. Joern + llvm2cpg + -fembed-bitcode -g -grecord-command-line -fno-inline-functions -fno-builtin
    - -g aby bylo možné zpětně namapovat LLVM IR/BC na zdrojový kód
    - zbytek je doporučeno přímo nástrojem llvm2cpg


#### Program slicing -- pomocí DG
  - DG je [nástroj](https://github.com/mchalupa/dg) pro analýzu kódu a program slicing
  - je k němu i [paper](https://www.fi.muni.cz/~xchalup4/dg_atva20_preprint.pdf)
  - návod na program slicing [zde](https://github.com/mchalupa/dg/blob/master/doc/llvm-slicer.md)
  - ```llvm-slicer``` vyžaduje, aby právě jeden z ```.bc``` souborů měl ```main``` --> 🔴to je problém🔴
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
  - 🔴DG opravdu nefunguje bez ```main``` fce🔴 --> nelze analyzovat knihovny



### Zmínit v textu
 1. [paper](https://ieeexplore.ieee.org/abstract/document/9376145?casa_token=AbkX5cmm18kAAAAA:oUjTofjHfN6VOcwFv1PoDWTm8Vr_rfqmoKwuwBNrFtYGMztIYH2HfhGG0rYTlgUVg7fZbkwL-A) o GNN nad Simplified CPG
 2. [studie](https://mediatum.ub.tum.de/doc/1659728/document.pdf) o chování statických analyzátorů nad syntetickými a reálnými benchamrky
 3. [studie](https://ieeexplore.ieee.org/abstract/document/9462962?casa_token=LZ2bQiYy1IgAAAAA:QrOOvx79MsJV0u9Vd4C9Dv4UGiSaFxfn-EDr0pWVH-wBhzW29b-s6DGS4cKJ9PPbYcjrpTGl3g) -- shallow vs deep learning pro detekci chyb
 4. [studie](https://dl.acm.org/doi/abs/10.1145/3338906.3338941) o perfektním labelování
 5. zkusit najít článek o porovnání úspěšnosti modelů na syntetických datasetech a reálných softwarech
