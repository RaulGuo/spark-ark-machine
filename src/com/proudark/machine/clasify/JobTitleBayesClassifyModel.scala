package com.proudark.machine.clasify

import org.apache.spark.sql.SparkSession
import com.proud.ark.config.ConfigUtil
import com.proud.ark.db.DBUtil
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import com.proudark.machine.util.TokenCNUtil
import com.proud.ark.data.HDFSUtil
import org.apache.spark.sql.SaveMode
import org.apache.spark.ml.feature.IDFModel
import org.apache.spark.sql.Row

/**
spark-shell --master spark://bigdata01:7077 --driver-memory 1g --executor-memory 13g --jars /home/data_center/dependency/mysql-connector-java.jar,/home/data_center/dependency/ArkUtil-0.0.1-SNAPSHOT.jar,/home/data_center/dependency/ArkMachine-0.0.1-SNAPSHOT.jar,/home/data_center/dependency/ikanalyzer-2012_u6.jar
spark-submit --master spark://bigdata01:7077 --driver-memory 1g --executor-memory 13g --class com.proudark.machine.clasify.JobTitleBayesClassifyModel /home/data_center/dependency/ArkMachine-0.0.1-SNAPSHOT-jar-with-dependencies.jar
 */

object JobTitleBayesClassifyModel {
  
  
  def initNaiveBayesModel(spark:SparkSession):(IDFModel, NaiveBayesModel) = {
    import spark.implicits._
    val df = DBUtil.loadDFFromTable("jobs.zhiwei_fenlei_training", spark)
    
    //1. 进行分词，输出字段tokens中是title_parsed的分词结果，是一个字符串的数组
    val tokenizer = new Tokenizer().setInputCol("title_parsed").setOutputCol("tokens")
    val tokenDF = tokenizer.transform(df)
    
    //2. 使用HashingTF计算Term Frequency。一个term在一个document中，出现了则tf为1，否则为0。
    val tf = getHashingTF()
    val tfDF = tf.transform(tokenDF)
    val idf = new IDF().setInputCol("term_freq").setOutputCol("idf")
    val idfModel = idf.fit(tfDF)
    val idfDF = idfModel.transform(tfDF)
    
    //NaiveBayes在train的时候对数据格式有要求，需要是RDD[LabeledPoint]类型的对象。
    //LabeledPoint对象中包括label和features两个字段，label是分类的结果，用double表示。feature是IDFModel计算得出的feature的tf-idf
    val metadata = idfDF.map(x => {
      val tfidf = x.getAs[org.apache.spark.ml.linalg.SparseVector]("idf").toArray
      LabeledPoint(x.getAs[Int]("fenlei_num"), Vectors.dense(tfidf))
    }).rdd
    
    val model:NaiveBayesModel = NaiveBayes.train(metadata)
    (idfModel, model)
  }
  
  def getHashingTF():HashingTF = {
    val tf = new HashingTF().setInputCol("tokens").setOutputCol("term_freq").setNumFeatures(2048*10)
    tf
  }
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master(ConfigUtil.master).appName("JobTitleBayesClassifyModel").config("spark.sql.warehouse.dir", ConfigUtil.warehouse).enableHiveSupport().getOrCreate()
    import spark.implicits._
    
//    val df = DBUtil.loadDFFromTable("jobs.zhiwei_fenlei_training", spark)
//    
//    //1. 进行分词，输出字段tokens中是title_parsed的分词结果，是一个字符串的数组
//    val tokenizer = new Tokenizer().setInputCol("title_parsed").setOutputCol("tokens")
//    val tokenDF = tokenizer.transform(df)
//    
//    //2. 使用HashingTF计算Term Frequency。一个term在一个document中，出现了则tf为1，否则为0。
//    val tf = new HashingTF().setInputCol("tokens").setOutputCol("term_freq").setNumFeatures(2048*10)
//    val tfDF = tf.transform(tokenDF)
//    
//    //3. 使用计算IDF，idf中包含是每个词的权重
//    val idf = new IDF().setInputCol("term_freq").setOutputCol("idf")
//    val idfModel = idf.fit(tfDF)
//    val idfDF = idfModel.transform(tfDF)
//    
//    //NaiveBayes在train的时候对数据格式有要求，需要是RDD[LabeledPoint]类型的对象。
//    //LabeledPoint对象中包括label和features两个字段，label是分类的结果，用double表示。feature是IDFModel计算得出的feature的tf-idf
//    val metadata = idfDF.map(x => {
//      val tfidf = x.getAs[org.apache.spark.ml.linalg.SparseVector]("idf").toArray
//      LabeledPoint(x.getAs[Int]("fenlei_num"), Vectors.dense(tfidf))
//    }).rdd
//    
//    val model:NaiveBayesModel = NaiveBayes.train(metadata)
    val (idfModel, model) = initNaiveBayesModel(spark)
    //定义执行分词的函数
    val tokenFunction:Function1[String, Array[String]] = TokenCNUtil.token
    import org.apache.spark.sql.functions._
    val tokenUdf = udf(tokenFunction)
    
    val dfToPredict = DBUtil.loadDFFromTable("jobs.company_employment", spark).select("id", "title").limit(100).withColumn("tokens", tokenUdf(col("title")))
    //NaiveBayesModel在执行predict的时候需要计算出一个document的tfidf，以向量的形式表示。
    val tf = getHashingTF()
    val tfDFToPred = tf.transform(dfToPredict)
    
    /**
     idfDFToPred.printSchema
      root
       |-- id: long (nullable = false)
       |-- title: string (nullable = true)
       |-- tokens: array (nullable = true)
       |    |-- element: string (containsNull = true)
       |-- term_freq: vector (nullable = true)
       |-- idf: vector (nullable = true)
     */
    //idfModel给DataFrame添加的字段的类型是
    val idfDFToPred = idfModel.transform(tfDFToPred)
    idfDFToPred.first().getAs[SparseVector]("idf")
    //定义执行将tf-idf转化为Vector的方法org.apache.spark.ml.linalg.SparseVector
    val predictFunc:Function1[org.apache.spark.ml.linalg.SparseVector, Double] = (vector:org.apache.spark.ml.linalg.SparseVector) => {
      val array = vector.toArray
      val v = Vectors.dense(array)
      model.predict(v)
    }
    val predictUdf = udf(predictFunc)
    
//    val result1 = idfDFToPred.map(r => {
//      Row.fromSeq(r.toSeq)
//    })
    
    val result = idfDFToPred.withColumn("fenlei_result", predictUdf(col("idf")))
    HDFSUtil.saveDSToHDFS("/home/mllib/job_classification", result, SaveMode.Overwrite)
    
  }
  
}