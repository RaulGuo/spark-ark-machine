package com.proudark.machine.test.linear

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.LinearRegression

object HousePriceTest {
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setAppName("SMS Message Classification (HAM or SPAM)")
    val spark = SparkSession.builder().master("local[*]").appName("Regression").config("spark.sql.warehouse.dir", "file:///root/spark-tmp/warehouse").enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._
    //三个字段分别代表总价、面积和卧室的数量
    val data = Array((498, 125, 3),(370, 87, 2), (498, 125, 3), (510, 116, 2), (1185, 350, 5))
    
    val df = spark.createDataFrame(data)
    
    val trainingSet = df.map { row => {
      val feature = row(0).asInstanceOf[Int]
      val vector = Vectors.dense(row(1).asInstanceOf[Int], row(2).asInstanceOf[Int])
      
      LabeledPoint(feature, vector)
    } }
    
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val lrModel = lr.fit(trainingSet)
    
    val modelSummary = lrModel.summary
    
    println(s"Cofficients: ${lrModel.coefficients}")
    println(s"intercept: ${lrModel.intercept}")
    println(s"totalIterations: ${modelSummary.totalIterations }")
    val resultDF = lrModel.transform(trainingSet)
    
    
  } 
}