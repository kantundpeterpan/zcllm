# LLM Zoomcamp - `dlt` and `cognee` Workshop


**Imports**

``` python
import dlt
import requests
```

**Ingestion helper function**

``` python
@dlt.resource
def zoomcamp_data():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            yield doc
```

# Question 1

``` python
!dlt --version
```

    dlt 1.14.1

# Question 2

``` python
from dlt.destinations import qdrant

qdrant_destination = qdrant(
  qd_path="db.qdrant", 
)
```

``` python
pipeline = dlt.pipeline(
    pipeline_name="zoomcamp_pipeline",
    destination=qdrant_destination,
    dataset_name="zoomcamp_tagged_data"

)
load_info = pipeline.run(zoomcamp_data())
print(pipeline.last_trace)
```

    /home/kantundpeterpan/projects/zoomcamp/zcllm/.venv/lib/python3.13/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

    Run started at 2025-07-30 08:34:03.976987+00:00 and COMPLETED in 21.69 seconds with 4 steps.
    Step extract COMPLETED in 1.00 seconds.

    Load package 1753864448.9672205 is EXTRACTED and NOT YET LOADED to the destination and contains no failed jobs

    Step normalize COMPLETED in 0.25 seconds.
    Normalized data for the following tables:
    - _dlt_pipeline_state: 1 row(s)
    - zoomcamp_data: 948 row(s)

    Load package 1753864448.9672205 is NORMALIZED and NOT YET LOADED to the destination and contains no failed jobs

    Step load COMPLETED in 15.46 seconds.
    Pipeline zoomcamp_pipeline load step completed in 15.42 seconds
    1 load package(s) were loaded to destination qdrant and into dataset zoomcamp_tagged_data
    The qdrant destination used /home/kantundpeterpan/projects/zoomcamp/zcllm/ws1/db.qdrant location to store data
    Load package 1753864448.9672205 is LOADED and contains no failed jobs

    Step run COMPLETED in 21.69 seconds.
    Pipeline zoomcamp_pipeline load step completed in 15.42 seconds
    1 load package(s) were loaded to destination qdrant and into dataset zoomcamp_tagged_data
    The qdrant destination used /home/kantundpeterpan/projects/zoomcamp/zcllm/ws1/db.qdrant location to store data
    Load package 1753864448.9672205 is LOADED and contains no failed jobs

``` python
!cat ./db.qdrant/meta.json 
```

    {"collections": {"zoomcamp_tagged_data": {"vectors": {"fast-bge-small-en": {"size": 384, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null, "strict_mode_config": null}, "zoomcamp_tagged_data_zoomcamp_data": {"vectors": {"fast-bge-small-en": {"size": 384, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null, "strict_mode_config": null}, "zoomcamp_tagged_data__dlt_version": {"vectors": {"fast-bge-small-en": {"size": 384, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null, "strict_mode_config": null}, "zoomcamp_tagged_data__dlt_loads": {"vectors": {"fast-bge-small-en": {"size": 384, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null, "strict_mode_config": null}, "zoomcamp_tagged_data__dlt_pipeline_state": {"vectors": {"fast-bge-small-en": {"size": 384, "distance": "Cosine", "hnsw_config": null, "quantization_config": null, "on_disk": null, "datatype": null, "multivector_config": null}}, "shard_number": null, "sharding_method": null, "replication_factor": null, "write_consistency_factor": null, "on_disk_payload": null, "hnsw_config": null, "wal_config": null, "optimizers_config": null, "init_from": null, "quantization_config": null, "sparse_vectors": null, "strict_mode_config": null}}, "aliases": {}}
