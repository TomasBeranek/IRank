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
    - [repo](https://github.com/Fraunhofer-AISEC/cpg) pro extrakci CPG z C/C++/Java (experimentálně pro Python, Golang, TypeScript)
      - má podporu i pro 🔴LLVM IR🔴❗ (zřejmě půjde napojit na Infer clang)
