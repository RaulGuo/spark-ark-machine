package com.proudark.machine.test

import org.apache.spark.sql.SparkSession
import org.wltea.analyzer.lucene.IKAnalyzer
import com.proudark.machine.util.TokenCNUtil

//import org.apache.spark.ml.feature.Word2Vec

object TokenTest {
  def main(args: Array[String]): Unit = {
    val content = "raul和齐达内是很牛逼的球星"
    println(TokenCNUtil.token(content))
  }
  
  
}