package com.proudark.machine.clasify

import org.apache.spark.sql.SparkSession
import com.proud.ark.config.ConfigUtil
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

object PerceptronTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("GuanliZhuanliToQiye").config("spark.sql.warehouse.dir", ConfigUtil.warehouse).getOrCreate()
    //libsvm的dataframe的格式为：label:Double features:Vector
    val data = spark.read.format("libsvm").load("/home/data_center/mllib/sample_multiclass_classification_data.txt")
    
    val Array(train,test) = data.randomSplit(Array(0.6, 0.4))
    
    //第一层是feature的数量，最后一层是分类结果的数量
    val layers = Array(4,5,6,4)
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers)
    
    //这里进行拟合时使用好像是LBFGS来求cost function的最小值，而不是Gradient Descent
    val model = trainer.fit(train)
    
    //对结果进行统计，result中包含原有的label、feature和新计算出来的predication字段
    val result = model.transform(test)
    
    
    
  }
}