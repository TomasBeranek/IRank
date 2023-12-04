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
    7. ```sudo rm /var/lib/neo4j/import/*``` -- smazání souborů z předchozího běhu
    8. ```sudo service neo4j status``` -- kontrola, jestli Neo4j běží
    9. ```sudo systemctl start neo4j.service``` -- pokud neběží, tak zapnout
    10. ```sudo cp /home/tomas/Documents/diplomka/code-extraction/example/out/*_data.csv /var/lib/neo4j/import/```
    11. ```find /home/tomas/Documents/diplomka/code-extraction/example/out -name 'nodes_*_cypher.csv' -exec /bin/cypher-shell -u neo4j -p 123 --file {} \;```
    12. ```find /home/tomas/Documents/diplomka/code-extraction/example/out -name 'edges_*_cypher.csv' -exec /bin/cypher-shell -u neo4j -p 123 --file {} \;```
    13. ```MATCH (n) RETURN n``` -- zobrazení v prohlížeči
    14. ```DETACH DELETE``` -- smazaní celé DB
    15. ```sudo rm /var/lib/neo4j/import/*``` -- vyčištění importu

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
  - 🔴lze si zobrazit graf v .dot🔴
  - pokud chci specifikovat i soubor, kde byla proměnná použita lze to pomocí ```-sc add_one.c##7#&p```, je to zmíněno [zde](https://github.com/mchalupa/dg/issues/350)


#### Generování grafů z datasetů
 - je nutné nainstalovat závislosti pro každý projekt - aby bylo možné spustit překlad požadovaných souborů

##### HTTPD
 - je nutné nahradit ```<$sys$>``` u překladových příkazů
 - v ```httpd/INSTALL``` uvadějí, že hlavní (a možna jedinou) závislostí je APACHE, čemuž by i odpovídal formát překladových příkazů ```<$sys$>/include/apr-1```
 - ziskání zdrojových souborů APACHE na Ubuntu
 1. je nutné stáhnout do složky ```<$repo$>srclib/``` knihovny ```apr``` a ```apr-util``` z https://apr.apache.org/download.cgi (fungující verze jsou ```apr-1.7.4``` a ```apr-util-1.6.3``` a přejmenovat je na ```apr``` a ```apr-util```
 2. jak knihovny, tak samotný projekt obsahuje ```.h.in``` šablony, ze kterých je nutno vygenerovat ```.h``` soubory - je nutno spustit ```<$repo$>/buildconf``` (vygeneruje ```configure```), poté ```<$repo$>/configure``` (vygeneruje ```.h``` soubory v repu i rekurzivně v knihovnách)
 3. nicméně soubor ```ap_config_auto.h``` je závislý na šabloně ```ap_config_auto.h.in``` a vždy, když se tato šablona změní v nějakém commitu oproti předchozímu, tak je nutno tento soubor přegenerovat
 4. dále je nutné nainstalovat pcre/pcre2 (některé z prnvích historických commitů již knihovnu obsahují, ale novější commity ne), pro instalaci ```sudo apt-get install libpcre3 libpcre3-dev``` (snad bude pcre3 zpětně kompatibilní), když je to nainstalováno manuálně, tak ./configure využije tuto verzi a nevyhodí error

##### NGINX
 - stačí pouze spustit ```./auto/configure``` pro vygenerování headerů

##### LIBTIFF
 - nutno nainstalovat OpenGL pomocí ```sudo apt install libgl-dev```, aby byl přítomen soubor ```GL/gl.h```
 - následně ještě GLUT pomocí ```sudo apt install freeglut3-dev```, pro soubor ```GL/glut.h```
 - poté ```./autogen.sh``` a poté ```./configure```

##### FFMPEG
 - nutno nainstalovat nasm pomocí ```sudo apt install nasm```
 - a yasm (potřeba pro nějaké commity např. 99d7a3e862ec0d15903f99f6b94d152ca1834f0f) pomocí ```sudo apt install yasm```
 - dále je potřeba nainstalovat SDL knihovny (kvůli SDL.h) pomocí ```sudo apt install libsdl2-dev```
 - poté ```./configure``` vygeneruje zejména ```config.h```

##### LIBAV
 - obsahuje řadu stejných souborů jako FFMPEG (tůzné knihovny pro zpracování videa/audia/obrázků)4
 - pro vytvoření ```config.h``` stačí ```./configure```

##### OPENSSL
 - Configure vyžaduje verzi Perlu 5.28.0 (aktuální 5.30.0 selže ve starších verzích ```2ac68bd6f1```), je možné využít perlbrew a spustit shell s 5.28.0 následovně ```perlbrew use perl-5.28.0``` (poté klasicky spustit pipeline)
 - pro vytvoření ```include/openssl/configuraion.h``` stačí ```./Configure gcc``` (v nových stačí pouze ./Configure, ale staré vyžadují specifikaci překladače, naštěstí nové jsou zpětně kompatibilní)

#### Experimenty s entry funkcí
  - mohou nastat v podstatě 3 případy chyb v kódu:
  1. scenario1 -- chyba začne v ```main``` a projeví se v ```f```
  2. scenario2 -- chyba začne v ```f``` a projeví s v ```main```
  2. scenario3 -- chyba začne v ```main``` a projeví se v ```main```, s tím, že ```f``` mělo nejaký vliv v průběhu
  - u všech scénářů je nutné považovat jako entry funkci ```main``` (či jinou fci), protože se jedná o nejvyšší funkci ve stromu volání a co je nad ní už nás nezajímá -- pro chybu to není podstatné (chyba nastane podle Inferu ikdyž vše co výše odstraníme)
  - experimenty ukázaly, že Infer nejvyšší funkci vždy uvede jako součást hlášení v poli ```procedure```

### Možná vylepšení
 - generovat .bc soubory clangem, který má Infer u sebe --> máme větší jistotu, že model uvidí to samé co Infer
 - udělat optimalizaci LLVM bitcode před llvm2cpg nebo ještě před LLVM-slicer -- mohlo by změnšit výstupní CPG
 - prořezávání pouze podle čísla řádku? -- může ušetřit spoustu problémů


### TODO
 - testy pro slicing criteria -- alespoň chování při prázdném, prázdném JSON ("[]") a plném soubory s reporty z Inferu
 - nějakým způsobem zakomponovat název chyby (nejdřív přeložit pomocí word2vec) do vstupu GNN -- node label?
 - oštřit, když pipeline najde .bc soubor, který ale neubsahuje bitcode je to nějaký jiný random formát
 - GNN6-GANZ -- někde získali BGNN4VD
 - problém s absencí Infer reportu u negativních vzorků museli také nějak ASI řešit u C-bert/D2A článku 🔴podívat se🔴
 - prozkoumat jak funguje LLVM-slicer -- podívat se vizuálně co odstraňuje (pomocí .dot formátu -- vše je zdokumentované [zde](https://github.com/mchalupa/dg/blob/master/doc/llvm-slicer.md#slicing-criteria)) a **zdokumentovat výsledky**
 - přidat timestamps do NOTE, WARNING a ERROR výpisů
 - v prezentaci označit, které části pipeline jsem já vytvořil (stačí hvězdičkou a dole v poznámce vysvětlit)
 - jak přidat více label = 0 vzorků? - vzít i verzi kódu "after" - po daném commitu mohl být kód změněn, ale label je stále 0

### Interesting
 - spuštění na LLVM-sliceru na combined LLVM bitcode může být pomalé, jelikož je daný bitcode velký (celý projekt), nicméně se to možná samo vyřeší díky specifikaci entry funkce, která to výrazně omezí
 - čistý Joern --> CPG o 998 uzlech, pomocí pipeline --> CPG o 103 uzlech -- otestovat a kdyžtak zmínit v textu


### Zmínit v textu
 1. [paper](https://ieeexplore.ieee.org/abstract/document/9376145?casa_token=AbkX5cmm18kAAAAA:oUjTofjHfN6VOcwFv1PoDWTm8Vr_rfqmoKwuwBNrFtYGMztIYH2HfhGG0rYTlgUVg7fZbkwL-A) o GNN nad Simplified CPG
 2. [studie](https://mediatum.ub.tum.de/doc/1659728/document.pdf) o chování statických analyzátorů nad syntetickými a reálnými benchamrky
 3. [studie](https://ieeexplore.ieee.org/abstract/document/9462962?casa_token=LZ2bQiYy1IgAAAAA:QrOOvx79MsJV0u9Vd4C9Dv4UGiSaFxfn-EDr0pWVH-wBhzW29b-s6DGS4cKJ9PPbYcjrpTGl3g) -- shallow vs deep learning pro detekci chyb
 4. [studie](https://dl.acm.org/doi/abs/10.1145/3338906.3338941) o perfektním labelování
 5. zkusit najít článek o porovnání úspěšnosti modelů na syntetických datasetech a reálných softwarech
 6. experiment potvrzující, že lze extrahovat entry funkci pro všechny možné případy extrahovat z Infer výstupu stejně -- viz. experimenty v ```entry-function-experiments/```
 7. uvést konkrétní příklad, kdy Infer detekuje chybu v podmíněném překladu (se 2 .h soubory) a Joern to nedokáže korektně namodelovat
 8. proč modely natrénované na syntetických datasetech nefungují na reálných programech? jednou z možností je podmíněný překlad
 9. ```-g``` při generování LLVM IR, je nutné pro použití criterií v potřebném formátu pro program slicing, více v [readme](https://github.com/mchalupa/dg/blob/master/doc/llvm-slicer.md#slicing-criteria)
 10. porovnání s chatGPT
 11. vylepšení oproti řešení modelů z IBM:
  - IBM nepoužívá program slicing program
  - nejsem si jist, zda vhodně označují místo výskytu chyby
 12. možné důvody proč to nefunguje
  - Infer občas chybně hlásí lokaci chyby -> D2A může být špatně označeno -> špatně se prořezává -> modelu chybí informace
  - u některých chyb Infer nehlásí bug trace a protože D2A s bug trace pracuje --> extrahuje z ní funkce, které jsou potřebné pro chybu, tak jelikož já závisím na tom samém, tak je možné, že za chybu mohou funkce, které nebyly v bug trace --> nebyly v D2A --> nebyly přeloženy --> nebyly z nich extrahovány CPG --> model nezná jejich kód (ikdyž by mohli být v data dependency grafu, kdyby byly překládány)
 13. extrakce slicing kriterií z různých typů chyb -- informace jsou uvedeny v slicing_criteria_extraction.py
 14. generování .bc souborů z .h souborů (viz. experiment) - dva typy 1) .h je pouze includnut 2) orezava se podle .h
 15. redukce velikosti datasetu pomoci symlinku (podivat se, kolik to usetrilo ```ls -lh | awk '{ sum += $5 } END { print "Total size: " sum }'``` - bez dereference symlinku, ```ls -lhL | awk '{ sum += $5 } END { print "Total size: " sum }'``` s dereferenci symlinku)
 16. při generování .bc souborů nutno ošetřit krajní připady jako např. při chybě httpd_17e63e9b25d3a852c6363cca0ae5e0d9bbdf028a_1, kde mají 2 překládané soubory stejný název - server/util.c a modules/dav/main/util.c
 17. urychlení hledání .bc souborů (viz. experiment/speed_test) - na openssl zrychlení cca 130x této části kódu v single thread módu (jenom toto hledání přídá jednotky hodin na celém datasetu)
 18. další krajní případ překladu vzorků - nepodporuji vzorky složeny pouze z .h (httpd: 8b2ec33ac5) souborů, protože z nich nelze vygenerovat .bc soubor - bylo by nutné najít soubour, který je includuje a ten přeložit 1) neznám překladové příkazy pro ty soubory 2) těchto vzorků je poměrně málo
 19. další krajní případ překladu vzorků - nepodporuji vzorky složené s neznámých .l a .y souborů 1) nejsou to typické zdrojové soubory a pochybuji, že si s tím llvm-slicer poradí (httpd: 8b2ec33ac5), opět jsou to pouze jednotky vzorků


#### Výsledky construction_phase_d2a.py
HTTPD_1 0/210 (failed/all) 0%
HTTPD_0 156/11974 (failed/all) ~1,3%
NGINX_1 1/418 (failed/all) ~0.2%
NGINX_0 37/17209 (failed/all) ~0.2%
LIBAV_1 83/4575 (failed/all) ~1.8%
LIBTIFF_1 0/534 (failed/all) 0%
LIBTIFF_0 8/11385 (failed/all) ~0.1%
OPENSSL_1 332/7913 (failed/all) ~4.2%
OPENSSL_0 30588/332584 (failed/all) ~9.2%
FFMPEG_1 150/4772 (failed/all) ~3.1%
