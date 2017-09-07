package com.proudark.machine.clasify

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import com.proud.ark.config.ConfigUtil

object LogisticRegressionTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ZhengliZhaopinXinxi").config("spark.sql.warehouse.dir", ConfigUtil.warehouse).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    
    val training = spark.read.format("libsvm").load("/home/data_center/mllib/sample_libsvm_data.txt")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    
  }
}