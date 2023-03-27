import numpy as np
from hops import hdfs, featurestore
import pyarrow as pa
import random
import pandas as pd
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import StructType, StructField, IntegerType

# IMPORTANT: must import  tensorflow before petastorm.tf_utils due to a bug in petastorm
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from pyspark.sql.types import StructType, StructField, IntegerType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.spark_utils import dataset_as_rdd
from petastorm import make_reader
from petastorm.tf_utils import tf_tensors, make_petastorm_dataset
from petastorm.pytorch import DataLoader
from petastorm import make_batch_reader


# The schema defines how the dataset schema looks like
HelloWorldSchema = Unischema('HelloWorldSchema', [
    UnischemaField('id', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('array_4d', np.uint8, (None, 128, 30, None), NdarrayCodec(), False),
])

HelloWorldSchema.as_spark_schema()