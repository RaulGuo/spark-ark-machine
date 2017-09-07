package com.proudark.machine.test

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.NaiveBayesModel
import com.proud.ark.config.ConfigUtil
import com.proud.ark.db.DBUtil
import com.proud.ark.data.HDFSUtil

object NaiveBayesTest {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master(ConfigUtil.master).appName("BayesModel").config("spark.sql.warehouse.dir", ConfigUtil.warehouse).enableHiveSupport().getOrCreate()
    import spark.implicits._
    
    val fenleiDF = DBUtil.loadDFFromTable("jobs.zhiwei_fenlei_training", spark)
    
//    val trainDF = sqlCtx.read.jdbc(DBUtil.db207url, "zhiwei_fenlei_training", DBUtil.prop).map(row => {
//      (row.getAs[Long]("id"), row.getAs[String]("title"), row.getAs[String]("fenlei"), row.getAs[Int]("fenlei_num"), row.getAs[String]("title_parsed").split(" "))
//    })
//    trainDF.show(100)
    //tokenizer用于处理分词，将一个字符串分词，转换成一个数组字段（Array[String]）
    val tokenizer = new Tokenizer().setInputCol("title_parsed").setOutputCol("title_token")
    
    val tokenDF = tokenizer.transform(fenleiDF)
    
//    tokenDF.show(100)
    
    //将数据分成训练数据和测试数据两部分
    val Array(trainDF, testDF) = tokenDF.randomSplit(Array(0.7, 0.3))
    //HashingTF用于将分词字段（也就是Array[String]）转换为一个向量对象，并计算器词频（TF）
    //OutputCol是一个向量对象，向量的维度是numFeatures，向量对象中记录的是该分词中的每个term位于feature中的哪个维度，以及term出现的frequency。
    //举一个向量的例子：[(2048,[73,947,1250,1708],[1.0,1.0,1.0,1.0])]
    //numFeatures表示将hash分桶的数量设置的个数。该数值越大，不同的词被算成一个hash值的概率就越小，数据也就越准确。
    val hashingTF = new HashingTF().setInputCol("title_token").setOutputCol("row_features").setNumFeatures(2048)
    
    //添加的词频统计的字段的内容格式：size为分词的数量，indices为分词的序号。value为1代表这几个序号的分词在这里都有
    //"row_features":{"type":0,"size":256,"indices":[13,92,172,232],"values":[1.0,1.0,1.0,1.0]}
    //一个term在资料库中出现的太频繁反而影响其重要性，因为这表示这个term没有特殊含义，需要使用IDF来衡量一个term提供了多少信息。
    val tfDF = hashingTF.transform(tokenDF)
    
//    tfDF.write.mode(SaveMode.Overwrite).json("F:\\MachineLearning\\tf")
    
    //计算TF-IDF,IDF的含义是inverse document frequency，计算的是
    //IDF的计算公式是：log(文档总数+1/DF(t)+1)，由于文档总数+1是一个固定的数，所以DF(t)+1越大，则该表达式的总数约小，约趋近于1。这样log表达式就趋近于0。
    //比较row_features和features(也就是IDF的输入和输出字段)：[(2048,[73,947,1250,1708],[1.0,1.0,1.0,1.0]),(2048,[73,947,1250,1708],[5.148462999891583,3.3567035306635287,5.841610180451529,3.20255285083627])]
    //可以理解，idf的输出字段中，包含了每个term的重要程度的信息。同样的term，其重要性在idf中是一样的。
    val idf = new IDF().setInputCol("row_features").setOutputCol("features")
    
    //训练出一个模型
    val idfModel = idf.fit(tfDF)
    
    //idf模型计算每个词频的idf的值。
    //features列的内容实例：
    //"features":{"type":0,"size":2048,"indices":[92,534,744,799,908,1293],
    //"values":[1.6931983969591524,4.375273111658102,1.9704091695436379,5.553928107999748,5.330784556685538,1.8839766637713304]}
    val tfidfDF = idfModel.transform(tfDF)
    
//    tfidfDF.write.mode(SaveMode.Overwrite).json("F:\\MachineLearning\\tfidf")
    
    tfidfDF.show(100)
    
    //将训练的数据转换成Bayes算法需要的格式。朴素贝叶斯算法中，需要的数据格式类似于：
    /**
     0, 1 0 1
     0, 1 3 2
     0, 3 0 0
     0, 4 0 0
     1, 0 1 0
     1, 0 2 0
     1, 0 3 0
     1, 0 4 0
     ....
     其中第一个数值是分类，用数字表示。后边的是feature的矩阵
     */
    
    //dense产生的是一个稠密矩阵，代表矩阵中每一行的每个值都会保存。这通常是浪费空间的方法。矩阵一行中包含多少列是由之前的hashingTF中的numFeatures确定的。
    //之前设置的是2048的话，这里一行中也有2048个数。
    val trainDataBayes = tfidfDF.select("fenlei_num", "features").map {
    x:Row => {
      val nums = x.get(1).asInstanceOf[org.apache.spark.ml.linalg.SparseVector].toArray
      LabeledPoint(x.getAs[Int]("fenlei_num"), Vectors.dense(nums))
      }
    }
//    trainDataBayes.write.mode(SaveMode.Overwrite).json("F:\\MachineLearning\\denseBayesTraining")
    
    //稀疏矩阵中则不会保存很多值，值保存不是0的点的值以及坐标。
    //使用稀疏矩阵的方法：
//val trainBayesSparse = tfidfDF.select("fenlei_num", "features").map{
//  case Row(label:Int, features:Vector) => {
//    LabeledPoint(label, features.toSparse)
//  }
//}

//    trainBayesSparse.write.mode(SaveMode.Overwrite).json("F:\\MachineLearning\\sparseBayesTraining")
    
    
    //训练模型：
    //lambda是平滑参数
    //modelType The type of NB model to fit from the enumeration NaiveBayesModels, can be multinomial or bernoulli
    val model:NaiveBayesModel = NaiveBayes.train(trainDataBayes.rdd, lambda=1.0, modelType="multinomial")
    
    model.save(spark.sparkContext, HDFSUtil.hdfsUrl+"/home/mllib/model")
    val sameModel = NaiveBayesModel.load(spark.sparkContext, HDFSUtil.hdfsUrl+"/home/mllib/model")
    
    //训练模型弄完了之后再继续用该模型
    val testTFDF = hashingTF.transform(testDF)
    val testTFIDFDF = idfModel.transform(testTFDF)
    val testTrainDataBayes = testTFIDFDF.select("fenlei_num", "features").map{
      x => {
        val nums = x.get(1).asInstanceOf[org.apache.spark.ml.linalg.SparseVector].toArray
        LabeledPoint(x.getAs[Int]("fenlei_num"), Vectors.dense(nums))
      }
    }

    val first = testTrainDataBayes.first().features


    testTrainDataBayes.foreach { x => println(model.predict(x.features)+"######"+x.label) }
    //对测试数据进行分类预测：
    val testpredictAndLabel = testTrainDataBayes.map(p => (model.predict(p.features), p.label))
    testpredictAndLabel.show(100)
    
    println("---------"+testpredictAndLabel.filter(x => x._1 == x._2).count())
    println(testpredictAndLabel.count())
    val accuracy = (testpredictAndLabel.filter(x => x._1 == x._2).count()*1.0)/testpredictAndLabel.count()
    
    println("准确率："+accuracy)
    
  }
  
}