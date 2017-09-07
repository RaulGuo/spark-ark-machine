package com.proudark.machine.test

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import breeze.linalg.sum

object LogisticRegressionTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SMS Message Classification (HAM or SPAM)")
    val spark = SparkSession.builder().master("local[*]").appName("DataImport").config("spark.sql.warehouse.dir", "file:///root/spark-tmp/warehouse").enableHiveSupport().getOrCreate()
    val sc = spark.sparkContext
    val sqlCtx = new SQLContext(sc)
    import sqlCtx.implicits._
    
    val training = sc.textFile("/home/data_center/mllib/hour.csv").map { x => x.split(",") }.cache()
    
    //yield生成的值会被记录下来，保存在集合中。循环结束后就返回该集合。
    val mapping = for(i <- Range(2, 10)) yield getMapping(training, i)
    
    val catLen = sum(mapping.map { x => x.size })
    
    
    
  }
  
  //将每个字段的值都保存下来
  def getMapping(rdd:RDD[Array[String]], idx:Int) = {
    val map = rdd.map { field => field(idx) }.distinct().zipWithIndex().collectAsMap()
    map
  }
  
}