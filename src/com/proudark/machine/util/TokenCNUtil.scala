package com.proudark.machine.util

import java.io.ByteArrayInputStream
import java.io.InputStreamReader
import org.wltea.analyzer.core.IKSegmenter
import org.wltea.analyzer.core.Lexeme

object TokenCNUtil {
  def token(content:String):Array[String] = {
    var set = Set[String]()
    val bt = content.getBytes;
    val inputStream = new ByteArrayInputStream(bt)
    val reader = new InputStreamReader(inputStream)
    val iksegmenter = new IKSegmenter(reader, false)
    println("-------------")
    
    var lexeme:Lexeme = null;
    while({lexeme = iksegmenter.next(); lexeme != null}){
      set = set+lexeme.getLexemeText
    }
    
    val it = Iterator.continually(iksegmenter.next()).takeWhile { _ != null }
    set.toArray
  }
  
  def main(args: Array[String]): Unit = {
    val content = "销售工程师"
    println(token(content))
  }
}