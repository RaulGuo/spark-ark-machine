package com.proudark.machine.test

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.regression.LinearRegressionSummary

object LinearRegressionTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SMS Message Classification (HAM or SPAM)")
    val spark = SparkSession.builder().master("local[*]").appName("DataImport").config("spark.sql.warehouse.dir", "file:///root/spark-tmp/warehouse").enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext
    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._
    
//    val training = MLUtils.loadLibSVMFile(sc, "/home/data_center/mllib/sample_linear_regression_data.txt")
    val training = spark.read.format("libsvm").load("/home/data_center/mllib/sample_linear_regression_data.txt")
    
    val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    
    val lrModel = lr.fit(training)
    val regressionSummary:LinearRegressionSummary = lrModel.evaluate(training)
    
    val result = lrModel.transform(training)
    
    val predictCol = regressionSummary.predictionCol
    val meanSqrErr = regressionSummary.meanSquaredError
    
    
    
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    
    val trainSummary = lrModel.summary
    println(s"numIterations: ${trainSummary.totalIterations}")
    println(s"objectiveHistory: [${trainSummary.objectiveHistory.mkString(",")}]")
    trainSummary.residuals.show()
    println(s"RMSE: ${trainSummary.rootMeanSquaredError}")
    println(s"r2: ${trainSummary.r2}")
    
  }
  
  def getMapping(rdd:RDD[Array[String]], idx:Int){
    rdd.map { field => field(idx) }.distinct().zipWithIndex().collectAsMap()
  }
}