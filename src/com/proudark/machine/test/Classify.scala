package com.proudark.machine.test

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.Tokenizer
import java.util.Properties

//import com.proudark.machine.util.DBUtil

object Classify {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("MachineLearning").master("local[*]").config("spark.sql.warehouse.dir", "file:///root/spark-tmp/warehouse").enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = new SQLContext(sc)
    
    val props = getProperties
    
    val fenciCol = "title_parsed"
    
//    val seedRdd = spark.read.jdbc("jdbc:mysql://192.168.1.207:3306/statistics?autoReconnect=true", "jobs.zhiwei_fenlei_seed", props).select(fenciCol, "fenlei_num").toDF("title", "fenlei_num")
//    .filter(row => row.getAs[String]("fenlei") != null).rdd.map { row => (row.getAs[String](fenciCol).split(" "), row.getAs[Long]("fenlei")) }
    
    val seedRdd = spark.read.jdbc("jdbc:mysql://192.168.1.207:3306/statistics?autoReconnect=true", "jobs.zhiwei_fenlei_seed", props).select(fenciCol, "fenlei")
    .filter(row => row.getAs[String](fenciCol) != null).rdd.map { row => {
      val content = row.getAs[String](fenciCol).split(",")
      (content(1).split(" "), row.getAs[String]("fenlei"))
    }}
    
    import sqlContext.implicits._
    
    val df = seedRdd.toDF("title", "fenlei")
    
    val labelIndexer = new StringIndexer().setInputCol("fenlei").setOutputCol("label").fit(df)
    
    val word2vec = new Word2Vec().setInputCol("title").setOutputCol("features")
    
    //神经网络的层数
    //输入层大小为4（features），两个媒介层（大小为5和4），输出层大小是3（classes）
    val layers = Array[Int](4, 5, 4, 3)
    
    val mlpc = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(512).setSeed(1234L).setMaxIter(128).setFeaturesCol("features")
    .setLabelCol("label").setPredictionCol("prediction")
    
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictLabel").setLabels(labelIndexer.labels)
    
    val dfArray = df.randomSplit(Array(0.8, 0.2))
    val trainingData = dfArray(0)
    val testData = dfArray(1)
    
    val stages = Array(labelIndexer, word2vec, mlpc, labelConverter)
//    val stages = Array(word2vec, mlpc)
    val pipeline = new Pipeline().setStages(stages)
    
    val model = pipeline.fit(trainingData)
    
    val predictionResultDF = model.transform(testData)
    
    predictionResultDF.printSchema()
    predictionResultDF.select("title", "fenlei", "predictLabel").show(100)
    
    val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("precision")
    
    val predictionAccuracy = evaluator.evaluate(predictionResultDF)
    
    println("Testing Accuracy is %2.4f".format(predictionAccuracy * 100) + "%")
    
    
    println("game over goodbye-----------------")
    
    val predictDF = spark.read.json("/root/SeedResult")
    predictDF.createOrReplaceTempView("predict")
    val allResult = spark.sql("select label,count(label) from predict group by label").toDF("label", "all")
    val wrongResult = spark.sql("select label,count(label) from predict where label != predictedLabel group by label").toDF("label", "wrong")
    val statisDF = allResult.join(wrongResult, "label")
    
    
    allResult.join(wrongResult, "label")
  }
  
  def getProperties(): Properties = {
	  var properties = new Properties()
	  properties.put("user", "root")
	  properties.put("password", "PS@Letmein123")
	  properties.put("driver", "com.mysql.jdbc.Driver")
	  properties.put("maxActive", "50")
	  properties.put("useServerPrepStmts", "false")
	  properties.put("rewriteBatchedStatements","true")
	  properties
  }
}