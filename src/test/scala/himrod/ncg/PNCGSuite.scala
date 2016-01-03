/*
 * Test suite for PNCG/ALS methods
 */

package himrod.ncg

import scala.collection.JavaConversions._
import scala.math.abs
import scala.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import org.jblas.DoubleMatrix

import org.scalatest.FunSuite

object NCGSuite {

  def generateRatingsAsJavaList(
      users: Int,
      products: Int,
      features: Int,
      samplingRate: Double,
      implicitPrefs: Boolean,
      negativeWeights: Boolean): (java.util.List[Rating[Int]], DoubleMatrix, DoubleMatrix) = {
    val (sampledRatings, trueRatings, truePrefs) =
      generateRatings(users, products, features, samplingRate, implicitPrefs)
    (seqAsJavaList(sampledRatings), trueRatings, truePrefs)
  }

  def generateRatings(
      users: Int,
      products: Int,
      features: Int,
      samplingRate: Double,
      implicitPrefs: Boolean = false,
      negativeWeights: Boolean = false,
      negativeFactors: Boolean = true): (Seq[Rating[Int]], DoubleMatrix, DoubleMatrix) = {
    val rand = new Random(42)

    // Create a random matrix with uniform values from -1 to 1
    def randomMatrix(m: Int, n: Int) = {
      if (negativeFactors) {
        new DoubleMatrix(m, n, Array.fill(m * n)(rand.nextDouble() * 2 - 1): _*)
      } else {
        new DoubleMatrix(m, n, Array.fill(m * n)(rand.nextDouble()): _*)
      }
    }

    val userMatrix = randomMatrix(users, features)
    val productMatrix = randomMatrix(features, products)
    val (trueRatings, truePrefs) = implicitPrefs match {
      case true =>
        // Generate raw values from [0,9], or if negativeWeights, from [-2,7]
        val raw = new DoubleMatrix(users, products,
          Array.fill(users * products)(
            (if (negativeWeights) -2 else 0) + rand.nextInt(10).toDouble): _*)
        val prefs =
          new DoubleMatrix(users, products, raw.data.map(v => if (v > 0) 1.0 else 0.0): _*)
        (raw, prefs)
      case false => (userMatrix.mmul(productMatrix), null)
    }

    val sampledRatings = {
      for (u <- 0 until users; p <- 0 until products if rand.nextDouble() < samplingRate)
        yield Rating(u, p, trueRatings.get(u, p).toFloat)
    }

    (sampledRatings, trueRatings, truePrefs)
  }
}


class NCGSuite extends FunSuite with LocalSparkContext {
/*class PNCGSuite extends SparkFunSuite with MLlibTestSparkContext {*/
    /*val (users,items) = PNCG.train(sampledRatings,features,10,10,iterations,0.01,implicitPrefs,1.0,false)*/

  val regParam = 1e-3
  val threshold = 0.1
  val users = 100
  val products = 300
  val rank = 15
  val samplingRate = 1.0
  val implicitPrefs = false
  val negativeWeights = false
  val negativeFactors = true
  val iters = 10
  /*val (sampledRatings, trueRatings, truePrefs) = NCGSuite.generateRatings(users, products,*/
  /*    rank, samplingRate, implicitPrefs, negativeWeights, negativeFactors)*/
  /*val R = sc.parallelize(sampledRatings)*/

  test("Testing ALS-NCG"){
    val (sampledRatings, trueRatings, truePrefs) = NCGSuite.generateRatings(users, products,
        rank, samplingRate, implicitPrefs, negativeWeights, negativeFactors)
    val R = sc.parallelize(sampledRatings)
    val (userFactors,itemFactors) = NCG.trainPNCG(R,rank,maxIter=iters,regParam=regParam)
    testRMSE(rank,trueRatings,truePrefs,userFactors,itemFactors,threshold,implicitPrefs)
  }
  test("Testing ALS"){
    val (sampledRatings, trueRatings, truePrefs) = NCGSuite.generateRatings(users, products,
        rank, samplingRate, implicitPrefs, negativeWeights, negativeFactors)
    val R = sc.parallelize(sampledRatings)
    val (userFactors,itemFactors) = NCG.trainALS(R,rank,maxIter=iters,regParam=regParam)
    testRMSE(rank,trueRatings,truePrefs,userFactors,itemFactors,threshold,implicitPrefs)
  }
  test("pseudorandomness") {
    val (sampledRatings, trueRatings, truePrefs) = NCGSuite.generateRatings(users, products,
        rank, samplingRate, implicitPrefs, negativeWeights, negativeFactors)
    val ratings = sc.parallelize(sampledRatings,2)

    /*val ratings = sc.parallelize(NCGSuite.generateRatings(10, 20, 5, 0.5, false, false)._1, 2)*/
    val model11 = NCG.trainPNCG(ratings,rank,maxIter=1,seed=1)
    val model12 = NCG.trainPNCG(ratings,rank,maxIter=1,seed=1)
    val u11 = model11._1.values.flatMap(_.toList).collect().toList
    val u12 = model12._1.values.flatMap(_.toList).collect().toList
    val model2 = NCG.trainPNCG(ratings,rank,maxIter=1,seed=2)
    val u2 = model2._1.values.flatMap(_.toList).collect().toList
    assert(u11 == u12)
    assert(u11 != u2)
  }

  test("Storage Level for RDDs in model") {
    val R = sc.parallelize(NCGSuite.generateRatings(10, 20, 5, 0.5, false, false)._1, 2)
    var storageLevel = StorageLevel.MEMORY_ONLY
    val (userFactors,itemFactors) = NCG.trainPNCG(R,rank,maxIter=1,regParam=regParam,finalRDDStorageLevel=storageLevel)
    assert(userFactors.getStorageLevel == storageLevel);
    assert(itemFactors.getStorageLevel == storageLevel);

    storageLevel = StorageLevel.DISK_ONLY
    val (ufac,ifac) = NCG.trainPNCG(R,rank,maxIter=1,regParam=regParam,finalRDDStorageLevel=storageLevel)
    assert(ufac.getStorageLevel == storageLevel);
    assert(ifac.getStorageLevel == storageLevel);
  }

  def testRMSE(
    rank: Int,
    trueRatings: DoubleMatrix,
    truePrefs: DoubleMatrix,
    userFactors: RDD[(Int, Array[Float])],
    itemFactors: RDD[(Int, Array[Float])],
    threshold: Double = 0.1,
    implicitPrefs: Boolean = false) = 
  {
    val model = new MatrixFactorizationModel(rank, 
      userFactors.mapValues{a => a.map{x => x.toDouble}}, 
      itemFactors.mapValues{a => a.map{x => x.toDouble}}
    )
    val users = userFactors.values.map{a => a.size / rank}.reduce{_+_}
    val products = itemFactors.values.map{a => a.size / rank}.reduce{_+_}

    val predictedU = new DoubleMatrix(users, rank)
    for ((u, vec) <- model.userFeatures.collect(); i <- 0 until rank) {
      predictedU.put(u, i, vec(i))
    }
    val predictedP = new DoubleMatrix(products, rank)
    for ((p, vec) <- model.productFeatures.collect(); i <- 0 until rank) {
      predictedP.put(p, i, vec(i))
    }
    val predictedRatings = {
      val allRatings = new DoubleMatrix(users, products)
      val usersProducts = for (u <- 0 until users; p <- 0 until products) yield (u, p)
      val userProductsRDD = sc.parallelize(usersProducts)
      model.predict(userProductsRDD).collect().foreach { elem =>
        allRatings.put(elem.user, elem.product, elem.rating)
      }
      allRatings
    }

    if (!implicitPrefs) {
      var sse = 0.0
      for (u <- 0 until users; p <- 0 until products) {
        val prediction = predictedRatings.get(u, p)
        val correct = trueRatings.get(u, p)
        val diff = (prediction - correct)
        sse += diff * diff
        if (math.abs(prediction - correct) > threshold) {
          fail(s"Model failed on: R_($u,$p) = $correct; predicted $prediction")
        }
      }
      println(s"rmse: ${math.sqrt(1.0/(users*products) * sse)}")
    } else {
      // For implicit prefs we use the confidence-weighted RMSE to test (ref Mahout's tests)
      var sqErr = 0.0
      var denom = 0.0
      for (u <- 0 until users; p <- 0 until products) {
        val prediction = predictedRatings.get(u, p)
        val truePref = truePrefs.get(u, p)
        val confidence = 1 + 1.0 * abs(trueRatings.get(u, p))
        val err = confidence * (truePref - prediction) * (truePref - prediction)
        sqErr += err
        denom += confidence
      }
      val rmse = math.sqrt(sqErr / denom)
      if (rmse > threshold) {
        fail(s"Model failed to predict RMSE: $rmse")
      }
    }
  }
}

