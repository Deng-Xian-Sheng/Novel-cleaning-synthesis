---
license: apache-2.0
base_model:
- Qwen/Qwen3-0.6B-Base
---
# Qwen3-Embedding-0.6B-GGUF

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="400"/>
<p>

## Highlights

The Qwen3 Embedding model series is the latest proprietary model of the Qwen family, specifically designed for text embedding and ranking tasks. Building upon the dense foundational models of the Qwen3 series, it provides a comprehensive range of text embeddings and reranking models in various sizes (0.6B, 4B, and 8B). This series inherits the exceptional multilingual capabilities, long-text understanding, and reasoning skills of its foundational model. The Qwen3 Embedding series represents significant advancements in multiple text embedding and ranking tasks, including text retrieval, code retrieval, text classification, text clustering, and bitext mining.

**Exceptional Versatility**: The embedding model has achieved state-of-the-art performance across a wide range of downstream application evaluations. The 8B size embedding model ranks **No.1** in the MTEB multilingual leaderboard (as of June 5, 2025, score **70.58**), while the reranking model excels in various text retrieval scenarios.

**Comprehensive Flexibility**: The Qwen3 Embedding series offers a full spectrum of sizes (from 0.6B to 8B) for both embedding and reranking models, catering to diverse use cases that prioritize efficiency and effectiveness. Developers can seamlessly combine these two modules. Additionally, the embedding model allows for flexible vector definitions across all dimensions, and both embedding and reranking models support user-defined instructions to enhance performance for specific tasks, languages, or scenarios.

**Multilingual Capability**: The Qwen3 Embedding series offer support for over 100 languages, thanks to the multilingual capabilites of Qwen3 models. This includes various programming languages, and provides robust multilingual, cross-lingual, and code retrieval capabilities.

## Model Overview

**Qwen3-Embedding-0.6B-GGUF** has the following features:

- Model Type: Text Embedding
- Supported Languages: 100+ Languages
- Number of Paramaters: 0.6B
- Context Length: 32k
- Embedding Dimension: Up to 1024, supports user-defined output dimensions ranging from 32 to 1024
- Quantization: q8_0, f16

For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our [blog](https://qwenlm.github.io/blog/qwen3-embedding/), [GitHub](https://github.com/QwenLM/Qwen3-Embedding).

## Qwen3 Embedding Series Model list

| Model Type       | Models               | Size | Layers | Sequence Length | Embedding Dimension | MRL Support | Instruction Aware |
|------------------|----------------------|------|--------|-----------------|---------------------|-------------|----------------|
| Text Embedding   | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 0.6B | 28     | 32K             | 1024                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)   | 4B   | 36     | 32K             | 2560                | Yes         | Yes            |
| Text Embedding   | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)   | 8B   | 36     | 32K             | 4096                | Yes         | Yes            |
| Text Reranking   | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B | 28     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)   | 4B   | 36     | 32K             | -                   | -           | Yes            |
| Text Reranking   | [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B)   | 8B   | 36     | 32K             | -                   | -           | Yes            |

> **Note**:
> - `MRL Support` indicates whether the embedding model supports custom dimensions for the final embedding. 
> - `Instruction Aware` notes whether the embedding or reranking model supports customizing the input instruction according to different tasks.
> - Our evaluation indicates that, for most downstream tasks, using instructions (instruct) typically yields an improvement of 1% to 5% compared to not using them. Therefore, we recommend that developers create tailored instructions specific to their tasks and scenarios. In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English.

## Usage

📌 **Tip**: We recommend that developers customize the `instruct` according to their specific scenarios, tasks, and languages. Our tests have shown that in most retrieval scenarios, not using an `instruct` on the query side can lead to a drop in retrieval performance by approximately 1% to 5%.

### llama.cpp
Check out our [llama.cpp documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html) for more usage guide.

We advise you to clone [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and install it following the official guide. We follow the latest version of llama.cpp. 
In the following demonstration, we assume that you are running commands under the repository `llama.cpp`.

You can run Qwen3 Embedding with one command:

```shell
./build/bin/llama-embedding -m model.gguf  -p "<your context here>"  --pooling last --verbose-prompt
```

Or launch a server:
```shell
./build/bin/llama-server -m model.gguf --embedding --pooling last -ub 8192 --verbose-prompt
```




## Evaluation

### MTEB (Multilingual)

| Model                            |  Size   |  Mean (Task)  | Mean (Type) | Bitxt Mining | Class. | Clust. | Inst. Retri. | Multi. Class. | Pair. Class. | Rerank | Retri. | STS  |
|----------------------------------|:-------:|:-------------:|:-------------:|:--------------:|:--------:|:--------:|:--------------:|:---------------:|:--------------:|:--------:|:--------:|:------:|
| NV-Embed-v2                      |   7B    |     56.29     | 49.58       | 57.84        | 57.29  | 40.80  | 1.04         | 18.63         | 78.94        | 63.82  | 56.72  | 71.10|
| GritLM-7B                        |   7B    |     60.92     | 53.74       | 70.53        | 61.83  | 49.75  | 3.45         | 22.77         | 79.94        | 63.78  | 58.31  | 73.33|
| BGE-M3                           |  0.6B   |     59.56     | 52.18       | 79.11        | 60.35  | 40.88  | -3.11        | 20.1          | 80.76        | 62.79  | 54.60  | 74.12|
| multilingual-e5-large-instruct   |  0.6B   |     63.22     | 55.08       | 80.13        | 64.94  | 50.75  | -0.40        | 22.91         | 80.86        | 62.61  | 57.12  | 76.81|
| gte-Qwen2-1.5B-instruct          |  1.5B   |     59.45     | 52.69       | 62.51        | 58.32  | 52.05  | 0.74         | 24.02         | 81.58        | 62.58  | 60.78  | 71.61|
| gte-Qwen2-7b-Instruct            |   7B    |     62.51     | 55.93       | 73.92        | 61.55  | 52.77  | 4.94         | 25.48         | 85.13        | 65.55  | 60.08  | 73.98|
| text-embedding-3-large           |    -    |     58.93     | 51.41       | 62.17        | 60.27  | 46.89  | -2.68        | 22.03         | 79.17        | 63.89  | 59.27  | 71.68|
| Cohere-embed-multilingual-v3.0   |    -    |     61.12     | 53.23       | 70.50        | 62.95  | 46.89  | -1.89        | 22.74         | 79.88        | 64.07  | 59.16  | 74.80|
| Gemini Embedding                 |    -    |     68.37     | 59.59       | 79.28        | 71.82  | 54.59  | 5.18         | **29.16**     | 83.63        | 65.58  | 67.71  | 79.40|
| **Qwen3-Embedding-0.6B**         |  0.6B   |     64.33     | 56.00       | 72.22        | 66.83  | 52.33  | 5.09         | 24.59         | 80.83        | 61.41  | 64.64  | 76.17|
| **Qwen3-Embedding-4B**           |   4B    |     69.45     | 60.86       | 79.36        | 72.33  | 57.15  | **11.56**    | 26.77         | 85.05        | 65.08  | 69.60  | 80.86|
| **Qwen3-Embedding-8B**           |   8B    |   **70.58**   | **61.69**   | **80.89**    | **74.00** | **57.65** | 10.06      | 28.66         | **86.40**    | **65.63** | **70.88** | **81.08** |

> **Note**: For compared models, the scores are retrieved from MTEB online [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) on May 24th, 2025.

### MTEB (Eng v2)

| MTEB English / Models          |  Param.  | Mean(Task) | Mean(Type) | Class. | Clust. | Pair Class. | Rerank. | Retri. | STS   | Summ. |
|--------------------------------|:--------:|:------------:|:------------:|:--------:|:--------:|:-------------:|:---------:|:--------:|:-------:|:-------:|
| multilingual-e5-large-instruct |   0.6B   | 65.53      | 61.21      | 75.54  | 49.89  | 86.24       | 48.74   | 53.47  | 84.72 | 29.89 |
| NV-Embed-v2                    |   7.8B   | 69.81      | 65.00      | 87.19  | 47.66  | 88.69       | 49.61   | 62.84  | 83.82 | 35.21 |
| GritLM-7B                      |   7.2B   | 67.07      | 63.22      | 81.25  | 50.82  | 87.29       | 49.59   | 54.95  | 83.03 | 35.65 |
| gte-Qwen2-1.5B-instruct        |   1.5B   | 67.20      | 63.26      | 85.84  | 53.54  | 87.52       | 49.25   | 50.25  | 82.51 | 33.94 |
| stella_en_1.5B_v5              |   1.5B   | 69.43      | 65.32      | 89.38  | 57.06  | 88.02       | 50.19   | 52.42  | 83.27 | 36.91 |
| gte-Qwen2-7B-instruct          |   7.6B   | 70.72      | 65.77      | 88.52  | 58.97  | 85.9        | 50.47   | 58.09  | 82.69 | 35.74 |
| gemini-embedding-exp-03-07     |    -     | 73.3       | 67.67      | 90.05  | 59.39  | 87.7        | 48.59   | 64.35  | 85.29 | 38.28 |
| **Qwen3-Embedding-0.6B**       |   0.6B   | 70.70      | 64.88      | 85.76  | 54.05  | 84.37       | 48.18   | 61.83  | 86.57 | 33.43 |
| **Qwen3-Embedding-4B**         |    4B    | 74.60      | 68.10      | 89.84  | 57.51  | 87.01       | 50.76   | 68.46  | 88.72 | 34.39 |
| **Qwen3-Embedding-8B**         |    8B    | 75.22      | 68.71      | 90.43  | 58.57  | 87.52       | 51.56   | 69.44  | 88.58 | 34.83 |

### C-MTEB (MTEB Chinese)

| C-MTEB           | Param. | Mean(Task) | Mean(Type) | Class. | Clust. | Pair Class. | Rerank. | Retr. | STS   |
|------------------|--------|------------|------------|--------|--------|-------------|---------|-------|-------|
| multilingual-e5-large-instruct | 0.6B   | 58.08      | 58.24      | 69.80  | 48.23  | 64.52       | 57.45   | 63.65 | 45.81 |
| bge-multilingual-gemma2 | 9B     | 67.64      | 75.31      | 59.30  | 86.67  | 68.28       | 73.73   | 55.19 | -     |
| gte-Qwen2-1.5B-instruct  | 1.5B   | 67.12      | 67.79      | 72.53  | 54.61  | 79.5        | 68.21   | 71.86 | 60.05 |
| gte-Qwen2-7B-instruct    | 7.6B   | 71.62      | 72.19      | 75.77  | 66.06  | 81.16       | 69.24   | 75.70 | 65.20 |
| ritrieve_zh_v1          | 0.3B   | 72.71      | 73.85      | 76.88  | 66.5   | 85.98       | 72.86   | 76.97 | 63.92 |
| **Qwen3-Embedding-0.6B** | 0.6B   | 66.33      | 67.45      | 71.40  | 68.74  | 76.42       | 62.58   | 71.03 | 54.52 |
| **Qwen3-Embedding-4B**   | 4B     | 72.27      | 73.51      | 75.46  | 77.89  | 83.34       | 66.05   | 77.03 | 61.26 |
| **Qwen3-Embedding-8B**   | 8B     | 73.84      | 75.00      | 76.97  | 80.08  | 84.23       | 66.99   | 78.21 | 63.53 |


## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}
```