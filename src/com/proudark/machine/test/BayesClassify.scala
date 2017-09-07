package com.proudark.machine.test

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import com.proud.ark.db.DBUtil
import com.proud.ark.config.ConfigUtil

object BayesClassify {
  
  case class RawDataRecord(category:String, text:String)
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("MachineLearning").master(ConfigUtil.master).config("spark.sql.warehouse.dir", ConfigUtil.warehouse).getOrCreate()
    val sc = spark.sparkContext
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    val props = Classify.getProperties
    
    val fenciCol = "title_parsed"
    
    val seedRdd = DBUtil.loadDFFromTable("jobs.zhiwei_fenlei_seed", spark).select("id", fenciCol, "fenlei_num").toDF("id", "title", "fenlei_num")
    .filter(row => row.getAs[String]("fenlei") != null).rdd.map { row => (row.getAs[String](fenciCol).split(" "), row.getAs[Long]("fenlei")) }
    
    val seedDF = seedRdd.toDF("titles", "fenlei")
    
    val Array(trainDF, testDF) = seedRdd.randomSplit(Array(0.7, 0.3))
    
    //计算分词的词频
    val hashingTF = new HashingTF().setNumFeatures(50000).setInputCol("titles").setOutputCol("tf")
    val featurizedData = hashingTF.transform(seedDF)
    featurizedData.printSchema()
    featurizedData.show(100)
    
    //根据TF计算IDF
    val idf = new IDF().setInputCol("tf").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    
    rescaledData.printSchema()
    rescaledData.show(100)
    
    //转换成Bayes的输入格式
    val trainDataDS = rescaledData.select("fenlei", "features").map{
      case Row(fenlei:String, features:Vector) => 
        LabeledPoint(fenlei.toDouble, Vectors.dense(features.toArray))
    }
    trainDataDS.printSchema()
    trainDataDS.show(100)
    
    //训练模型
//    val model = NaiveBayes.train(input)
    
    
  }
}