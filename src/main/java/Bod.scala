/**
  * Created by itim on 24.02.2018.
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Bod {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    val IsNight = udf {
      (hour: String) =>
        var h = hour.toInt
        if (h > 5 && h < 18) {
          0
        } else {
          1
        }
    }
    var weekend = Array("Saturday", "Sunday")
    val IsWeekend = udf {
      (x: String) =>
        if (weekend.contains(x)) {
          1
        } else {
          0
        }
    }
    val data = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv("data/train1.csv").select("Dates", "Category", "DayOfWeek", "PdDistrict", "Address").filter(x => (!x.anyNull))
    println(data.select("Address").distinct().count())
    var data2 = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv("data/test.csv").select("Dates", "DayOfWeek", "PdDistrict", "Address").filter(x => (!x.anyNull))
    //print(data.select("Category").distinct().count())
    val Categories = Array(
      "Category", "DayOfWeek",
      "PdDistrict", "Address")
    data.printSchema()

    var IndexedCategories = new Array[StringIndexer](4)
    for (i <- 0 to 3) {
      IndexedCategories(i) = new StringIndexer()
        .setInputCol(Categories(i))
        .setOutputCol("Indexed" + Categories(i))
    }
    var pipeline = new Pipeline().setStages(IndexedCategories)
    val res = pipeline.fit(data).transform(data)
    data2 = IndexedCategories(3).fit(data2).transform(data2)
    data2 = IndexedCategories(2).fit(data2).transform(data2)
    data2 = IndexedCategories(1).fit(data2).transform(data2)
    data2.show()
    var df = res
      .withColumn("HourOfDay", hour(col("Dates")))
      .withColumn("Month", month(col("Dates")))
      .withColumn("Year", year(col("Dates")))
      .withColumn("IsNight", IsNight(col("HourOfDay")))
      .withColumn("IsWeekend", IsWeekend(col("DayOfWeek")))
    //DayOfWeek PdDistrict Address DayOrNight
    var assembler = new VectorAssembler()
      .setInputCols(Array("IndexedDayOfWeek", "IndexedPdDistrict", "HourOfDay", "Month", "Year"))
      .setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("IndexedCategory").setOutputCol("label")
    df = labelIndexer.fit(df).transform(df)
    df = assembler.transform(df)
    val splits = df.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    val numClasses = 39
    val impurity = "gini"
    val maxDepth = 10
    val maxBins = 32
    val classifier = new RandomForestClassifier()
      .setImpurity(impurity)
      .setMaxDepth(maxDepth)
      .setNumTrees(10)
      .setFeatureSubsetStrategy("all")
      .setSeed(1034)
      .setMaxMemoryInMB(1024)
    val model = classifier.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.show()
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
    println("Accuracy = " + accuracy)
  }
}
