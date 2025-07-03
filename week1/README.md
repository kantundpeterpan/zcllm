# LLM: Week 1 - Introduction
Heiner Atze

# Introdution to LLM and RAG

# Retrieval and Search

# Generation using OpenAI

# Clean the code

# Replace the toy search engine with Elasticsearch

- run Elastic search in Docker

``` bash
docker run -it \
    --rm \
    --name elasticsearch \
    -m 4GB \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

## Index the documents with ElasticSearch

- persistent indexing
