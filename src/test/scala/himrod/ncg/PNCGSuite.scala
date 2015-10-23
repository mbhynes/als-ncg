/*
 * Test suite for PNCG/ALS methods
 */

package himrod.ncg

import scala.collection.JavaConversions._
import scala.math.abs
import scala.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel

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
  val users = 500
  val products = 100
  val rank = 10
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
  /*test("NCG, f=10"){*/
  /*  val (sampledRatings, trueRatings, truePrefs) = NCGSuite.generateRatings(users, products,*/
  /*      rank, samplingRate, implicitPrefs, negativeWeights, negativeFactors)*/
  /*  val R = sc.parallelize(sampledRatings)*/
  /*  val (userFactors,itemFactors) = NCG.trainNCG(R,rank,maxIter=iters,regParam=regParam)*/
  /*  testRMSE(rank,trueRatings,truePrefs,userFactors,itemFactors,threshold,implicitPrefs)*/
  /*}*/

  /*test("rank-1 matrices") {*/
  /*  testPNCG(50, 100, 1, 15, 0.7, 0.3)*/
  /*}*/

  /*test("rank-1 matrices bulk") {*/
  /*  testPNCG(50, 100, 1, 15, 0.7, 0.3, false, true)*/
  /*}*/

  /*test("rank-2 matrices") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.3)*/
  /*}*/

  /*test("rank-2 matrices bulk") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.3, false, true)*/
  /*}*/

  /*test("rank-1 matrices implicit") {*/
  /*  testPNCG(80, 160, 1, 15, 0.7, 0.4, true)*/
  /*}*/

  /*test("rank-1 matrices implicit bulk") {*/
  /*  testPNCG(80, 160, 1, 15, 0.7, 0.4, true, true)*/
  /*}*/

  /*test("rank-2 matrices implicit") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.4, true)*/
  /*}*/

  /*test("rank-2 matrices implicit bulk") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.4, true, true)*/
  /*}*/

  /*test("rank-2 matrices implicit negative") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.4, true, false, true)*/
  /*}*/

  /*test("rank-2 matrices with different user and product blocks") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.4, numUserBlocks = 4, numProductBlocks = 2)*/
  /*}*/

  /*test("pseudorandomness") {*/
  /*  val ratings = sc.parallelize(PNCGSuite.generateRatings(10, 20, 5, 0.5, false, false)._1, 2)*/
  /*  val model11 = PNCG.train(ratings, 5, 1, 1.0, 2, 1)*/
  /*  val model12 = PNCG.train(ratings, 5, 1, 1.0, 2, 1)*/
  /*  val u11 = model11.userFeatures.values.flatMap(_.toList).collect().toList*/
  /*  val u12 = model12.userFeatures.values.flatMap(_.toList).collect().toList*/
  /*  val model2 = PNCG.train(ratings, 5, 1, 1.0, 2, 2)*/
  /*  val u2 = model2.userFeatures.values.flatMap(_.toList).collect().toList*/
  /*  assert(u11 == u12)*/
  /*  assert(u11 != u2)*/
  /*}*/

  /*test("Storage Level for RDDs in model") {*/
  /*  val ratings = sc.parallelize(PNCGSuite.generateRatings(10, 20, 5, 0.5, false, false)._1, 2)*/
  /*  var storageLevel = StorageLevel.MEMORY_ONLY*/
  /*  var model = new PNCG()*/
  /*    .setRank(5)*/
  /*    .setIterations(1)*/
  /*    .setLambda(1.0)*/
  /*    .setBlocks(2)*/
  /*    .setSeed(1)*/
  /*    .setFinalRDDStorageLevel(storageLevel)*/
  /*    .run(ratings)*/
  /*  assert(model.productFeatures.getStorageLevel == storageLevel);*/
  /*  assert(model.userFeatures.getStorageLevel == storageLevel);*/
  /*  storageLevel = StorageLevel.DISK_ONLY*/
  /*  model = new PNCG()*/
  /*    .setRank(5)*/
  /*    .setIterations(1)*/
  /*    .setLambda(1.0)*/
  /*    .setBlocks(2)*/
  /*    .setSeed(1)*/
  /*    .setFinalRDDStorageLevel(storageLevel)*/
  /*    .run(ratings)*/
  /*  assert(model.productFeatures.getStorageLevel == storageLevel);*/
  /*  assert(model.userFeatures.getStorageLevel == storageLevel);*/
  /*}*/

  /*test("negative ids") {*/
  /*  val data = PNCGSuite.generateRatings(50, 50, 2, 0.7, false, false)*/
  /*  val ratings = sc.parallelize(data._1.map { case Rating(u, p, r) =>*/
  /*    Rating(u - 25, p - 25, r)*/
  /*  })*/
  /*  val correct = data._2*/
  /*  val model = PNCG.train(ratings, 5, 15)*/

  /*  val pairs = Array.tabulate(50, 50)((u, p) => (u - 25, p - 25)).flatten*/
  /*  val ans = model.predict(sc.parallelize(pairs)).collect()*/
  /*  ans.foreach { r =>*/
  /*    val u = r.user + 25*/
  /*    val p = r.product + 25*/
  /*    val v = r.rating*/
  /*    val error = v - correct.get(u, p)*/
  /*    assert(math.abs(error) < 0.4)*/
  /*  }*/
  /*}*/

  /*test("NNPNCG, rank 2") {*/
  /*  testPNCG(100, 200, 2, 15, 0.7, 0.4, false, false, false, -1, -1, false)*/
  /*}*/

  /**
   * Test if we can correctly factorize R = U * P where U and P are of known rank.
   *
   * @param users number of users
   * @param products number of products
   * @param features number of features (rank of problem)
   * @param iterations number of iterations to run
   * @param samplingRate what fraction of the user-product pairs are known
   * @param matchThreshold max difference allowed to consider a predicted rating correct
   * @param implicitPrefs flag to test implicit feedback
   * @param bulkPredict flag to test bulk predicition
   * @param negativeWeights whether the generated data can contain negative values
   * @param numUserBlocks number of user blocks to partition users into
   * @param numProductBlocks number of product blocks to partition products into
   * @param negativeFactors whether the generated user/product factors can have negative entries
   */
  // scalastyle:off
  /*def runTest(*/
  /*    users: Int,*/
  /*    products: Int,*/
  /*    features: Int,*/
  /*    iterations: Int,*/
  /*    samplingRate: Double,*/
  /*    matchThreshold: Double,*/
  /*    implicitPrefs: Boolean = false,*/
  /*    bulkPredict: Boolean = false,*/
  /*    negativeWeights: Boolean = false,*/
  /*    numUserBlocks: Int = -1,*/
  /*    numProductBlocks: Int = -1,*/
  /*    negativeFactors: Boolean = true) {*/
    // scalastyle:on

    /*val (sampledRatings, trueRatings, truePrefs) = PNCGSuite.generateRatings(users, products,*/
      /*features, samplingRate, implicitPrefs, negativeWeights, negativeFactors)*/
    /*val (users,items) = PNCG.train(sampledRatings,features,10,10,iterations,0.01,implicitPrefs,1.0,false)*/
    /*val R = sc.parallelize(sampledRatings)*/
    /*val (userFactors,itemFactors) = PNCG.train(R,features,maxIter=iterations) //0.01,implicitPrefs,1.0,false)*/

    /*val model = new PNCG()*/
    /*  .setUserBlocks(numUserBlocks)*/
    /*  .setProductBlocks(numProductBlocks)*/
    /*  .setRank(features)*/
    /*  .setIterations(iterations)*/
    /*  .setAlpha(1.0)*/
    /*  .setImplicitPrefs(implicitPrefs)*/
    /*  .setLambda(0.01)*/
    /*  .setSeed(0L)*/
    /*  .setNonnegative(!negativeFactors)*/
    /*  .run(sc.parallelize(sampledRatings))*/
/**/
/*    */
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
        /*println(s"R_($u,$p) = $correct; predicted $prediction")*/
        if (math.abs(prediction - correct) > threshold) {
          fail(s"Model failed on: R_($u,$p) = $correct; predicted $prediction")
          /*fail(("Model failed to predict (%d, %d): %f vs %f\ncorr")*/
          /*  .format(u, p, correct, prediction))*/
          /*fail(("Model failed to predict (%d, %d): %f vs %f\ncorr: %s\npred: %s\nU: %s\n P: %s")*/
          /*  .format(u, p, correct, prediction, trueRatings, predictedRatings, predictedU,*/
              /*predictedP))*/
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
        /*fail("Model failed to predict RMSE: %f\ncorr: %s\npred: %s\nU: %s\n P: %s".format(*/
        /*  rmse, truePrefs, predictedRatings, predictedU, predictedP))*/
        fail(s"Model failed to predict RMSE: $rmse")
      }
    }
  }
}

