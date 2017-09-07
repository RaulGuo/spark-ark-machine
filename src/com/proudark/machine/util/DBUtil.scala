package com.proudark.machine.util

import java.util.Properties

object DBUtil {
  val db207url = "jdbc:mysql://192.168.1.207:3306/jobs?autoReconnect=true"
  
  def getProperties(): Properties = {
	  var properties = new Properties()
	  properties.put("user", "root")
	  properties.put("password", "PS@Letmein123")
	  properties.put("driver", "com.mysql.jdbc.Driver")
	  properties.put("maxActive", "50")
	  properties.put("useServerPrepStmts", "false")
	  properties.put("rewriteBatchedStatements","true")
	  properties
  }
  
  val prop = getProperties;
}