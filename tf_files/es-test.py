
# coding: utf-8

# In[2]:


# coding: utf-8
import os
import sys

from io import StringIO
from IPython.display import Image, display, clear_output
from elasticsearch import Elasticsearch

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
    'content': 'Test Message',
    'text': 'Can you hear me now?',
    'number': x
    }
    es_post = es.index(index="test", doc_type="_doc", body=tdoc)
    print('ES document sent.')
    print(tdoc)

