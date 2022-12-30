MASTER = "local"
NUM_PROCESSORS = "8"
NUM_EXECUTORS = "4"
NUM_PARTITIONS = 10

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import pyspark.sql.functions as psf
import json
import sys

conf = SparkConf()

conf.set("spark.app.name", "one_part_data")
conf.set("spark.master", MASTER)
conf.set("spark.executor.cores", NUM_PROCESSORS)
conf.set("spark.executor.instances", NUM_EXECUTORS)
conf.set("spark.executor.memory", "6g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "2000")
conf.set("spark.executor.heartbeatInterval", "6000s")
conf.set("spark.network.timeout", "10000000s")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.driver.memory", "15g")
conf.set("spark.driver.maxResultSize", "15g")

from pyspark.sql import SparkSession

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Create SparkSession 
spark = SparkSession.builder \
      .config(conf=conf) \
      .master("local[*]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()

class Train():
    """
        Сlass that allows you to train Kmeans models, save them for future use
    """
    @staticmethod
    def load_and_preprocess_data_drom_csv(name_file="/home/yagor/Рабочий стол/mipt/lab3/notebook/data_from_hbase_with_postprocess.csv"):
        """
                Class method which load and preprocess data for train Kmeans
            Args:
                name_file str: file to data in .csv format
            Returns:
                sparkDataset: preprocess data for train Kmeans
        """
        dataset = spark.read.csv(name_file,header=True,inferSchema=True)
        feat_cols = [ #'_c0',
            'fat_100g',
            'carbohydrates_100g',
            'sugars_100g',
            'proteins_100g',
            'salt_100g',
            'energy_100g',
            'reconstructed_energy',
            'g_sum',
            'exceeded',
            #'product'
            ]  # columns for train model (all without index and string data)
        vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')
        final_data = vec_assembler.transform(dataset)
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
        # Compute summary statistics by fitting the StandardScaler
        scalerModel = scaler.fit(final_data)
        # Normalize each feature to have unit standard deviation.
        cluster_final_data = scalerModel.transform(final_data)
        return cluster_final_data
    
    @staticmethod
    def train_model(cluster_final_data, k=8):
        """
                Class method which train and predict clusters with KMeans model
            Args:
                cluster_final_data sparkDataset: data for clusterization
                k int: count means in KMeans
            Returns:
                result: sparkDataset in format c_0, prediction, where c_0 - index, prediction - predict cluster
        """
        kmeans = KMeans(featuresCol='scaledFeatures',k=k)
        model = kmeans.fit(cluster_final_data)
        predictions = model.transform(cluster_final_data)
        cols = ["_c0", "prediction"]
        result = predictions.select(*cols)
        return result

    @staticmethod
    def save_results(result, name_file = "output_result.csv"):
        """
                Class method which save predict clusters with indexs
            Args:
                result sparkDataset: format c_0, prediction, where c_0 - index, prediction - predict cluster
                name_file str: name file for saves results (.csv)
        """
        result.write.format("com.databricks.spark.csv") \
            .option("header", "true") \
            .save(name_file)

if __name__ == "__main__":
    try:
        name_file = str(sys.argv[1])
        k = int(sys.argv[2])
        file_for_save_result = str(sys.argv[3])
    except:
        print('Введите корректные входные данные')
        sys.exit(1)
    # Create SparkSession
    spark = SparkSession.builder \
      .config(conf=conf) \
      .master("local[*]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()
    cluster_final_data = Train.load_and_preprocess_data_drom_csv(name_file)
    result = Train.train_model(cluster_final_data, k)
    Train.save_results(result, name_file = file_for_save_result)
