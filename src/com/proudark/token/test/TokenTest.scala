package com.proudark.token.test

import org.wltea.analyzer.lucene.IKAnalyzer
import java.io.StringReader
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import java.util.HashMap

object TokenTest {
  def main(args: Array[String]): Unit = {
    val text = "lxw的大数据田地 -- lxw1234.com 专注Hadoop、Spark、Hive等大数据技术博客。 北京优衣库";
    val analyzer = new IKAnalyzer(true)
    val reader = new StringReader(text)
    val tokenStream = analyzer.tokenStream("", reader);
    
    val terms  = tokenStream.getAttribute(classOf[CharTermAttribute])
    tokenStream.reset
    while(tokenStream.incrementToken()){
      println(terms.toString()+"|")
    }
    
    analyzer.close()
    reader.close()
  }
}