# eventcause
A neural network with event-related features for causal relation identification in discourse. This code implements the architecture proposed in the paper:

Edoardo Maria Ponti, Anna Korhonen. 2017. **[Event-related features in feedforward neural networks contribute to identifying causal relations in discourse](http://www.aclweb.org/anthology/W17-0903).** In *Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics*, pp. 25-30.

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{ponti2017event,
  title={Event-related features in feedforward neural networks contribute to identifying causal relations in discourse},
  author={Ponti, Edoardo Maria and Korhonen, Anna},
  booktitle={Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics},
  pages={25--30},
  year={2017}
}
```

## Requirements

- Theano

## Setup: datasets and word embeddings

- Download the relevant data: [Penn Discourse TreeBank](https://www.seas.upenn.edu/~pdtb/) and [CSTNews Corpus](nilc.icmc.usp.br/CSTNews/);

- Download the pickled files for English and Portuguese word embeddings from [Polyglot](https://sites.google.com/site/rmyeid/projects/polyglot)

- Preprocess the data in order to obtain the following tabular format (distances are between each token and the two event mentions):

| left sentence | right sentence | left event | right event | left arguments | right arguments | left distances | right distances | numerical label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
*they sow a row of male-fertile plants nearby then* | *which pollinate the male-sterile plants* | sow | pollinate | they row | which plants | 1-10\|0-9\|1-8\|2-7\|3-6\|4-5\|5-4\|6-3\|7-2 | 8-1\|9-0\|10-1\|11-2\|12-3	| 0

## Execution

The train the baseline (ignoring the additional features) launch ```mlp-base-RstDT.py```, for the proposed model ```mlp-feature-RstDT.py```. The model also allows for synthetic data generation and under/over-sampling. The results reported in the paper are:

| Classifier | Macro-F1 | Precision | Recall |
| --- | --- | --- | --- |
| Positive | 42.11 | 26.67 | 100 |
| Basic | 53.01 | 42.04 | 71.74 | 66.44 |
| Feature | 54.52 | 42.37 | 76.45 | 66.35 |

## License

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
