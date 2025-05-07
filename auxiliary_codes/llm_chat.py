from llm import llm


query_ = """
You are dealing with a small heterogeneous knowledge graph of nodes and edges.
You have to answer the question based on the graph.
Each node in the graph contains a text description, and the edge represents the relationship between nodes and the direction of information transmission.
The graph is:
Number of paper nodes: 3.
Number of author nodes: 3.

 Paper nodes:
『**Target** Paper node 0』, description: paper_id is: 6, year is: 2008, title is: Unsupervised feature selection for principal components analysis, abstract is: Principal Components Analysis (PCA) is the predominant linear dimensionality reduction technique, and has been widely applied on datasets in all scientific domains. We consider, both theoretically and empirically, the topic of unsupervised feature selection for PCA, by leveraging algorithms for the so-called Column Subset Selection Problem (CSSP). In words, the CSSP seeks the "best" subset of exactly  k  columns from an  m  x  n  data matrix A, and has been extensively studied in the Numerical Linear Algebra community. We present a novel two-stage algorithm for the CSSP. From a theoretical perspective, for small to moderate values of  k , this algorithm significantly improves upon the best previously-existing results [24, 12] for the CSSP. From an empirical perspective, we evaluate this algorithm as an unsupervised feature selection strategy in three application domains of modern statistical data analysis: finance, document-term data, and genetics. We pay particular attention to how this algorithm may be used to select representative or landmark features from an object-feature matrix in an unsupervised manner. In all three application domains, we are able to identify  k  landmark features, i.e., columns of the data matrix, that capture nearly the same amount of information as does the subspace that is spanned by the top  k  "eigenfeatures.",

『Paper node 1』, description: paper_id is: 1329, year is: 2006, title is: Tensor-CUR decompositions for tensor-based data, abstract is: Motivated by numerous applications in which the data may be modeled by a variable subscripted by three or more indices, we develop a tensor-based extension of the matrix CUR decomposition. The tensor-CUR decomposition is most relevant as a data analysis tool when the data consist of one mode that is qualitatively different than the others. In this case, the tensor-CUR decomposition approximately expresses the original data tensor in terms of a basis consisting of underlying subtensors that are actual data elements and thus that have natural interpretation in terms ofthe processes generating the data. In order to demonstrate the general applicability of this tensor decomposition, we apply it to problems in two diverse domains of data analysis: hyperspectral medical image analysis and consumer recommendation system analysis. In the hyperspectral data application, the tensor-CUR decomposition is used to  compress  the data, and we show that classification quality is not substantially reduced even after substantial data compression. In the recommendation system application, the tensor-CUR decomposition is used to  reconstruct  missing entries in a user-product-product preference tensor, and we show that high quality recommendations can be made on the basis of a small number of basis users and a small number of product-product comparisons from a new user.,

『Paper node 2』, description: paper_id is: 10202, year is: 2006, title is: Matrix approximation and projective clustering via volume sampling, abstract is: Frieze et al. [17] proved that a small sample of rows of a given matrix  A  contains a low-rank approximation  D  that minimizes || A - D || F  to within small additive error, and the sampling can be done efficiently using just two passes over the matrix [12]. In this paper, we generalize this result in two ways. First, we prove that the additive error drops exponentially by iterating the sampling in an adaptive manner. Using this result, we give a pass-efficient algorithm for computing low-rank approximation with reduced additive error. Our second result is that using a natural distribution on subsets of rows (called  volume  sampling), there exists a subset of  k  rows whose span contains a factor ( k  + 1) relative approximation and a subset of  k  +  k ( k  + 1)/&epsilon; rows whose span contains a 1+&epsilon; relative approximation. The existence of such a small certificate for multiplicative low-rank approximation leads to a PTAS for the following projective clustering problem: Given a set of points  P  in R  d  , and integers  k, j , find a set of  j  subspaces  F  1 , . . .,  F   j  , each of dimension at most  k , that minimize &Sigma;  p &isin;P min  i    d(p, F   i  ) 2 .,


 Author nodes:
『Author node 0』, description: author_id is: 384, name is: Petros Drineas, firm is: ['Yahoo! Research', 'Rensselaer Polytechnic Institute', 'Yale University'].

『Author node 1』, description: author_id is: 9884, name is: Michael W. Mahoney, firm is: ['Yahoo! Research', 'Rensselaer Polytechnic Institute', 'Stanford University', 'Yale University'].

『Author node 2』, description: author_id is: 12147, name is: Christos Boutsidis, firm is: ['Rensselaer Polytechnic Institute'].


 Edges:
『paper node 1』 -- cited_by --> 『paper node 0』
『paper node 2』 -- cited_by --> 『paper node 0』
『author node 0』 -- writes --> 『paper node 0』
『author node 1』 -- writes --> 『paper node 0』
『author node 2』 -- writes --> 『paper node 0』

Question: Which conference is 『**Target** Paper node 0』 published in? Give 5 likely conferences from ['KDD', 'CIKM', 'WWW', 'SIGIR', 'STOC', 'MobiCOMM', 'SIGMOD', 'SIGCOMM', 'SPAA', 'ICML', 'VLDB', 'SOSP', 'SODA', 'COLT'].
Answer format is like: [conference1, conference2, conference3, conference4, conference5].
And give your reason for the answer.
Answer:
Reason:
"""

query_ = """
You are dealing with a small heterogeneous knowledge graph of nodes and edges.
You have to answer the question based on the graph.
Each node in the graph contains a text description, and the edge represents the relationship between nodes and the direction of information transmission.
The graph is:
Number of paper nodes: 6.
Number of author nodes: 4.

 Paper nodes:
『**Target** Paper node 0』, description: paper_id is: 9, year is: 2008, title is: Structured learning for non-smooth ranking losses, abstract is: Learning to rank from relevance judgment is an active research area. Itemwise score regression, pairwise preference satisfaction, and listwise structured learning are the major techniques in use. Listwise structured learning has been applied recently to optimize important non-decomposable ranking criteria like AUC (area under ROC curve) and MAP (mean average precision). We propose new, almost-linear-time algorithms to optimize for two other criteria widely used to evaluate search systems: MRR (mean reciprocal rank) and NDCG (normalized discounted cumulative gain) in the max-margin structured learning framework. We also demonstrate that, for different ranking criteria, one may need to use different feature maps. Search applications should not be optimized in favor of a single criterion, because they need to cater to a variety of queries. E.g., MRR is best for navigational queries, while NDCG is best for informational queries. A key contribution of this paper is to fold multiple ranking loss functions into a multi-criteria max-margin optimization. The result is a single, robust ranking model that is close to the best accuracy of learners trained on individual criteria. In fact, experiments over the popular LETOR and TREC data sets show that, contrary to conventional wisdom, a test criterion is often not best served by training with the same individual criterion.,

『Paper node 1』, description: paper_id is: 1318, year is: 2006, title is: Training linear SVMs in linear time, abstract is: Linear Support Vector Machines (SVMs) have become one of the most prominent machine learning techniques for high-dimensional sparse data commonly encountered in applications like text classification, word-sense disambiguation, and drug design. These applications involve a large number of examples  n  as well as a large number of features  N , while each example has only  s  &lt;&lt;  N  non-zero features. This paper presents a Cutting Plane Algorithm for training linear SVMs that provably has training time  0(s,n)  for classification problems and  o ( sn  log ( n ))for ordinal regression problems. The algorithm is based on an alternative, but equivalent formulation of the SVM optimization problem. Empirically, the Cutting-Plane Algorithm is several orders of magnitude faster than decomposition methods like svm light for large datasets.,

『Paper node 2』, description: paper_id is: 5559, year is: 2000, title is: IR evaluation methods for retrieving highly relevant documents, abstract is: This paper proposes evaluation methods based on the use of non-dichotomous relevance judgements in IR experiments. It is argued that evaluation methods should credit IR methods for their ability to retrieve highly relevant documents. This is desirable from the user point of view in modern large IR environments. The proposed methods are (1) a novel application of P-R curves and average precision computations based on separate recall bases for documents of different degrees of relevance, and (2) two novel measures computing the cumulative gain the user obtains by examining the retrieval result up to a given ranked position. We then demonstrate the use of these evaluation methods in a case study on the effectiveness of query types, based on combinations of query structures and expansion, in retrieving documents of various degrees of relevance. The test was run with a best match retrieval system (In-Query 1 ) in a text database consisting of newspaper articles. The results indicate that the tested strong query structures are most effective in retrieving highly relevant documents. The differences between the query types are practically essential and statistically significant. More generally, the novel evaluation methods and the case demonstrate that non-dichotomous relevance assessments are applicable in IR experiments, may reveal interesting phenomena, and allow harder testing of IR methods.,

『Paper node 3』, description: paper_id is: 6875, year is: 2007, title is: A support vector method for optimizing average precision, abstract is: Machine learning is commonly used to improve ranked retrieval systems. Due to computational difficulties, few learning techniques have been developed to directly optimize for mean average precision (MAP), despite its widespread use in evaluating such systems. Existing approaches optimizing MAP either do not find a globally optimal solution, or are computationally expensive. In contrast, we present a general SVM learning algorithm that efficiently finds a globally optimal solution to a straightforward relaxation of MAP. We evaluate our approach using the TREC 9 and TREC 10 Web Track corpora (WT10g), comparing against SVMs optimized for accuracy and ROCArea. In most cases we show our method to produce statistically significant improvements in MAP scores.,

『Paper node 4』, description: paper_id is: 6889, year is: 2007, title is: Frank: a ranking method with fidelity loss, abstract is: Ranking problem is becoming important in many fields, especially in information retrieval (IR). Many machine learning techniques have been proposed for ranking problem, such as RankSVM, RankBoost, and RankNet. Among them, RankNet, which is based on a probabilistic ranking framework, is leading to promising results and has been applied to a commercial Web search engine. In this paper we conduct further study on the probabilistic ranking framework and provide a novel loss function named fidelity loss for measuring loss of ranking. The fidelity loss notonly inherits effective properties of the probabilistic ranking framework in RankNet, but possesses new properties that are helpful for ranking. This includes the fidelity loss obtaining zero for each document pair, and having a finite upper bound that is necessary for conducting query-level normalization. We also propose an algorithm named FRank based on a generalized additive model for the sake of minimizing the fedelity loss and learning an effective ranking function. We evaluated the proposed algorithm for two datasets: TREC dataset and real Web search dataset. The experimental results show that the proposed FRank algorithm outperforms other learning-based ranking methods on both conventional IR problem and Web search.,

『Paper node 5』, description: paper_id is: 8676, year is: 2002, title is: Optimizing search engines using clickthrough data, abstract is: This paper presents an approach to automatically optimizing the retrieval quality of search engines using clickthrough data. Intuitively, a good information retrieval system should present relevant documents high in the ranking, with less relevant documents following below. While previous approaches to learning retrieval functions from examples exist, they typically require training data generated from relevance judgments by experts. This makes them difficult and expensive to apply. The goal of this paper is to develop a method that utilizes clickthrough data for training, namely the query-log of the search engine in connection with the log of links the users clicked on in the presented ranking. Such clickthrough data is available in abundance and can be recorded at very low cost. Taking a Support Vector Machine (SVM) approach, this paper presents a method for learning retrieval functions. From a theoretical perspective, this method is shown to be well-founded in a risk minimization framework. Furthermore, it is shown to be feasible even for large sets of queries and features. The theoretical results are verified in a controlled experiment. It shows that the method can effectively adapt the retrieval function of a meta-search engine to a particular group of users, outperforming Google in terms of retrieval quality after only a couple of hundred training examples.,


 Author nodes:
『Author node 0』, description: author_id is: 5862, name is: Soumen Chakrabarti, firm is: ['Indian Institute of Technology Bombay', 'IBM', 'University of California', 'Carnegie Mellon University'].

『Author node 1』, description: author_id is: 13163, name is: Rajiv Khanna, firm is: ['Indian Institute of Technology Bombay', 'Yahoo! Research'].

『Author node 2』, description: author_id is: 13183, name is: Chiru Bhattacharyya, firm is: [].

『Author node 3』, description: author_id is: 13235, name is: Uma Sawant, firm is: ['Indian Institute of Technology Bombay'].


 Edges:
『paper node 1』 -- cited_by --> 『paper node 0』
『paper node 2』 -- cited_by --> 『paper node 0』
『paper node 3』 -- cited_by --> 『paper node 0』
『paper node 4』 -- cited_by --> 『paper node 0』
『paper node 5』 -- cited_by --> 『paper node 0』
『author node 0』 -- writes --> 『paper node 0』
『author node 1』 -- writes --> 『paper node 0』
『author node 2』 -- writes --> 『paper node 0』
『author node 3』 -- writes --> 『paper node 0』
Question: Which conference is 『**Target** Paper node 0』 published in? Give 5 likely conferences from ['KDD', 'CIKM', 'WWW', 'SIGIR', 'STOC', 'MobiCOMM', 'SIGMOD', 'SIGCOMM', 'SPAA', 'ICML', 'VLDB', 'SOSP', 'SODA', 'COLT'].
Answer format is like: [conference1, conference2, conference3, conference4, conference5].
And give your reason for the answer.
Answer:
Reason:
"""


from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer

chat_memory = ChatMemoryBuffer.from_defaults(token_limit=300000)
chat_engine = SimpleChatEngine.from_defaults(
    memory=chat_memory,
    llm=llm
)

response = chat_engine.chat(query_)
print(response)

next_msg = input("Please enter your next message: ")
while next_msg.lower() != "exit":
    response = chat_engine.chat(next_msg)
    print(response)
    next_msg = input("Please enter your next message: ")
