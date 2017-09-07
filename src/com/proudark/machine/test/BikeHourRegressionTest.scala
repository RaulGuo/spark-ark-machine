package com.proudark.machine.test

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import breeze.linalg.sum
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Row
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegressionSummary

object BikeHourRegressionTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SMS Message Classification (HAM or SPAM)")
    val spark = SparkSession.builder().master("local[*]").appName("Regression").config("spark.sql.warehouse.dir", "file:///root/spark-tmp/warehouse").enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext

//    val records2 = spark.read.format("libsvm").load("/home/data_center/mllib/hour.csv")
//    val training = MLUtils.loadLibSVMFile(sc, "/home/data_center/mllib/sample_linear_regression_data.txt")
    
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    
    val recordsRdd = sc.textFile("/home/data_center/mllib/hour.csv").map { x => x.split(",")}
    
    val data = recordsRdd.map { row => {
      val label = row(16).toDouble
      val vector = Vectors.dense(row(2).toDouble,row(3).toDouble,row(4).toDouble,row(5).toDouble,row(6).toDouble
          ,row(7).toDouble,row(8).toDouble,row(9).toDouble,row(10).toDouble,row(11).toDouble,row(12).toDouble,
          row(13).toDouble,row(14).toDouble,row(15).toDouble)
      
      LabeledPoint(label, vector)
    } }.toDF()
    
    val Array(training, evaluate) = data.randomSplit(Array(0.7, 0.4))
    
    
    //regParam代表正则化参数，
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val lrModel = lr.fit(training)
    lrModel.setPredictionCol("predict_col")
    
    val testSummary:LinearRegressionSummary = lrModel.evaluate(evaluate)
    
    testSummary.coefficientStandardErrors
    
    
//    //2到10个特征都是类别特征，不是实数特征
//    val mappings=for(i<-Range(2,10)) yield getMapping(records,i)
//
//    val catLen=sum(mappings.map(_.size))
//    val numLen=records.first().slice(10,14).size
//    val total_len=catLen+numLen
  }
  
  val dataFormat = new SimpleDateFormat("yyyy-MM-dd")
  
  def toDate(str:String):Date =
  {
    dataFormat.parse(str)
  }
}