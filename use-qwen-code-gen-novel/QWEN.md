你当前在一个docker容器里面，容器创建时的网络模式是host，所以我在容器外可以访问容器内启动的服务，并且容器内外共享65535个端口。

我在做这样的事儿：通过 https://chat.dphn.ai 提供的API + milvus + Qwen3-Embedding-0.6B + OpenWebUI 实现一个写小说的AI助手。

事情是这样的，https://chat.dphn.ai 提供了一个HTTP接口，有点像OpenAI 的 /v1/chat/completions API，但不是，只是有一点点类似。

这个API提供的模型比较弱，所以这个模型的能力有限。

openai_proxy_prod.py 里面基于https://chat.dphn.ai 提供的API做了一个封装，它能开放出一个更像OpenAI 的 /v1/chat/completions API的HTTP服务。

我们也许能用上openai_proxy_prod.py，也许需要做些改变，我不会让https://chat.dphn.ai 提供的模型调用工具，它没那么厉害。我会尽可能的减轻模型的负担。

milvus运行在容器外，我启动时的配置文件如下，这有助于你了解端口：
```
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.25
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2024-12-18T13-15-44Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.6.12
    command: ["milvus", "run", "standalone"]
    security_opt:
    - seccomp:unconfined
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MQ_TYPE: woodpecker
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```

Qwen3-Embedding-0.6B，我没有GPU，我以GGUF的形式用llama.cpp跑的，也是在容器外，我这样跑的：
```
./llama-b7770/llama-server -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --embedding --pooling last -ub 8192 --verbose-prompt --port 8081
```

OpenWebUI还没有，你可以clone一个仓库，这个我计划在容器里面，也就是你这里，我们大概需要改它。

对于容器外的服务，有问题你可以告诉我，让我来操作

如果我们使用openai_proxy_prod.py，openai_proxy_prod.py也会在容器里运行。

/workspace这个目录是容器外映射到容器内的。我这个容器是一个相对复杂的Dockerfile构建的，它是一个开发环境。

对于我们要做的事儿，当前现状和我的想法是：
现在milvus和Qwen3-Embedding-0.6B已经就绪，我随时可以运行它们。
OpenWebUI还没有，我认为可能，需要改它。
如何对接https://chat.dphn.ai 是用openai_proxy_prod.py还是改改用openai_proxy_prod.py还是根据openai_proxy_prod.py分析https://chat.dphn.ai 接口的逻辑然后专门写一个用于小说助手的对接逻辑。你得从openai_proxy_prod.py 分析https://chat.dphn.ai 接口逻辑了，它比我告诉你更全面。
milvus现在是空的，你得根据当前的小说目录结构写个脚本将小说向量化并存进去。


存放小说的目录是这样的：
在write_novel/novel，目录结构是：
```
novel/                          # 小说根目录
├── novel1/                          # 第1个小说目录
│   ├── 用户消息.md                   # 生成这个小说时的要求，例如“请你写一个……”。如果用户的要求和这个相似，则考虑参考这个小说。
│   ├── 综述.md                       # 小说的综述
│   ├── 正文.md                       # 有的小说没有分段落，这是没分段落时的正文
│   │
│   ├── {段落标题}-block1/            # 第1段目录（有段落时）
│   │   ├── 段落综述.md               # 这个段落的综述（最后一段没有）
│   │   └── 正文.md                   # 段落正文内容
│   │
│   ├── {段落标题}-block2/            # 第2段目录
│   │   ├── 段落综述.md
│   │   └── 正文.md
│   │
│   └── {段落标题}-blockN/            # 第N段目录
│       ├── 段落综述.md
│       └── 正文.md
│
├── novel2/                          # 第2个小说目录
│   └── ...
│
└── novelN/                          # 第N个小说目录
    └── ...
```
我不希望你列出或者读取这个目录，因为这会额外占用你的上下文，然后没什么用。

dataset.jsonl 这个也不要读取。

Qwen3-Embedding-0.6B-HF-README.md 这个里面的内容有助于你实现：请求llama.cpp提供的服务将小说向量化到milvus。将用户查询向量化并查询milvus的逻辑。
主要是这个模型有两种向量，一个问题一个文档，还有一个指令用于调整向量化的偏好什么的。

Qwen3-Embedding-0.6B-GGUF-HF-README.md 这个文档基本上是如何跑GGUF的东西，对你来说应该没用。

qwen-code 这个目录与我们要做的事儿完全无关。

生成小说助手提示词时的回答.md 顾名思义，我早些时间用gpt5.4 high生成了一些用于小说助手的提示词，供你借鉴。

生成openai_proxy_prod.py时的回答.md 这个是用gpt5.4 high生成openai_proxy_prod.py时的回答，包含了一些openai_proxy_prod.py的用法、依赖、代码逻辑什么的。

openai_proxy_prod.py 由我来运行，你只需要告诉我怎么做。（如果用openai_proxy_prod.py的话）

我能想到最基础的是：
将用户要求向量化搜库 -> 将用户要求和搜库结果给 https://chat.dphn.ai -> 给出小说文本
或者
根据用户要求请求一次https://chat.dphn.ai 让它编写搜库的文本，然后向量化搜库 -> 将用户要求和搜库结果给 https://chat.dphn.ai -> 给出小说文本
你可以结合我给你的信息思考更好的实现。

为了避免你搞出问题，任何大改动的方案都得先告诉我详细的过程，让我看看，我再决定是否做。

我可能会用好几个上下文/会话来做完这件事儿，为了你能了解进度，你可以创建一个进度.md，随时更新进度，这样当我新开会话与你交流的时候，你就了解进度了。

write_novel/QWEN.md 里面是旧的东西，几乎没啥用。