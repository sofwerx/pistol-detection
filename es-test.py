# coding: utf-8
<<<<<<< HEAD
from elasticsearch import Elasticsearch
from datetime import datetime
=======
from IPython.display import Image, display, clear_output
from elasticsearch import Elasticsearch
>>>>>>> aef14fd0b98d4520689d5b31359f38ebd698eea4

# Hold warnings
import warnings
warnings.filterwarnings('ignore')

# Setup ES 
try:
    es = Elasticsearch(
        [
            'https://elastic:diatonouscoggedkittlepins@elasticsearch.orange.opswerx.org:443'
        ],
        verify_certs=True
    )
    print("ES - Connected.")
except Exception as ex:
    print("Error: ", ex)

# Send results to ES
for x in range(5):
    tdoc = {
<<<<<<< HEAD
    'timestamp': datetime.now(),
    'content': 'Test Message',
    'text': 'Can you hear me now?',
=======
    'content': 'Test Message',
    'text': 'Can you hear me now?',
    'dude': 'GC',
>>>>>>> aef14fd0b98d4520689d5b31359f38ebd698eea4
    'number': x,
    }
    es_post = es.index(index="test", doc_type="_doc", body=tdoc)
    print('ES document sent.')
    print(tdoc)
