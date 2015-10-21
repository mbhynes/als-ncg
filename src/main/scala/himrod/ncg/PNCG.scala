/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package himrod.ncg

import himrod.ncg.utils._

import java.util.Random
import java.{util => ju}
import java.io.IOException

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Sorting
import scala.util.hashing.byteswap64

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.netlib.util.intW

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.{Logging, Partitioner}
/*import org.apache.spark.annotation.{DeveloperApi, Experimental}*/
/*import org.apache.spark.ml.{Estimator, Model}*/
/*import org.apache.spark.ml.param._*/
/*import org.apache.spark.ml.param.shared._*/
/*import org.apache.spark.ml.util.{Identifiable, SchemaUtils}*/
/*import org.apache.spark.mllib.optimization.NNLS*/
import org.apache.spark.rdd.RDD
/*import org.apache.spark.sql.DataFrame*/
/*import org.apache.spark.sql.functions._*/
/*import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructType}*/
import org.apache.spark.storage.StorageLevel
/*import org.apache.spark.util.Utils*/
/*import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet, SortDataFormat, Sorter}*/

/*import org.apache.spark.util.random.XORShiftRandom*/

/**
 * :: Highly Experimental ::
 * Nonlinearly preconditioned CG Accelerated Alternating Least Squares (ALS) matrix factorization.
 *
 * ALS attempts to estimate the ratings matrix `R` as the product of two lower-rank matrices,
 * `X` and `Y`, i.e. `X * Yt = R`. Typically these approximations are called 'factor' matrices.
 * The general approach is iterative. During each iteration, one of the factor matrices is held
 * constant, while the other is solved for using least squares. The newly-solved factor matrix is
 * then held constant while solving for the other factor matrix.
 *
 * This is a blocked implementation of the ALS factorization algorithm that groups the two sets
 * of factors (referred to as "users" and "products") into blocks and reduces communication by only
 * sending one copy of each user vector to each product block on each iteration, and only for the
 * product blocks that need that user's feature vector. This is achieved by pre-computing some
 * information about the ratings matrix to determine the "out-links" of each user (which blocks of
 * products it will contribute to) and "in-link" information for each product (which of the feature
 * vectors it receives from each user block it will depend on). This allows us to send only an
 * array of feature vectors between each user block and product block, and have the product block
 * find the users' ratings and update the products based on these messages.
 *
 * For implicit preference data, the algorithm used is based on
 * "Collaborative Filtering for Implicit Feedback Datasets", available at
 * [[http://dx.doi.org/10.1109/ICDM.2008.22]], adapted for the blocked approach used here.
 *
 * Essentially instead of finding the low-rank approximations to the rating matrix `R`,
 * this finds the approximations for a preference matrix `P` where the elements of `P` are 1 if
 * r > 0 and 0 if r <= 0. The ratings then act as 'confidence' values related to strength of
 * indicated user
 * preferences rather than explicit ratings given to items.
 */
/*@Experimental*/
/*class PNCG(override val uid: String) extends Estimator[ALSModel] with ALSParams {*/
/**/
/*  /*import org.apache.spark.ml.recommendation.PNCG.Rating*/*/
/**/
/*  def this() = this(Identifiable.randomUID("als"))*/
/**/
/*  /** @group setParam */*/
/*  def setRank(value: Int): this.type = set(rank, value)*/
/**/
/*  /** @group setParam */*/
/*  def setNumUserBlocks(value: Int): this.type = set(numUserBlocks, value)*/
/**/
/*  /** @group setParam */*/
/*  def setNumItemBlocks(value: Int): this.type = set(numItemBlocks, value)*/
/**/
/*  /** @group setParam */*/
/*  def setImplicitPrefs(value: Boolean): this.type = set(implicitPrefs, value)*/
/**/
/*  /** @group setParam */*/
/*  def setAlpha(value: Double): this.type = set(alpha, value)*/
/**/
/*  /** @group setParam */*/
/*  def setUserCol(value: String): this.type = set(userCol, value)*/
/**/
/*  /** @group setParam */*/
/*  def setItemCol(value: String): this.type = set(itemCol, value)*/
/**/
/*  /** @group setParam */*/
/*  def setRatingCol(value: String): this.type = set(ratingCol, value)*/
/**/
/*  /** @group setParam */*/
/*  def setPredictionCol(value: String): this.type = set(predictionCol, value)*/
/**/
/*  /** @group setParam */*/
/*  def setMaxIter(value: Int): this.type = set(maxIter, value)*/
/**/
/*  /** @group setParam */*/
/*  def setRegParam(value: Double): this.type = set(regParam, value)*/
/**/
/*  /** @group setParam */*/
/*  def setNonnegative(value: Boolean): this.type = set(nonnegative, value)*/
/**/
/*  /** @group setParam */*/
/*  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)*/
/**/
/*  /** @group setParam */*/
/*  def setSeed(value: Long): this.type = set(seed, value)*/
/**/
/*  /***/
/*   * Sets both numUserBlocks and numItemBlocks to the specific value.*/
/*   * @group setParam*/
/*   */*/
/*  def setNumBlocks(value: Int): this.type = {*/
/*    setNumUserBlocks(value)*/
/*    setNumItemBlocks(value)*/
/*    this*/
/*  }*/
/**/
/*  override def fit(dataset: DataFrame): ALSModel = {*/
/*    import dataset.sqlContext.implicits._*/
/*    val ratings = dataset*/
/*      .select(col($(userCol)).cast(IntegerType), col($(itemCol)).cast(IntegerType),*/
/*        col($(ratingCol)).cast(FloatType))*/
/*      .map { row =>*/
/*        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))*/
/*      }*/
/*    val (users, items) = PNCG.train(ratings, rank = $(rank),*/
/*      numUserBlocks = $(numUserBlocks), numItemBlocks = $(numItemBlocks),*/
/*      maxIter = $(maxIter), regParam = $(regParam), implicitPrefs = $(implicitPrefs),*/
/*      alpha = $(alpha), nonnegative = $(nonnegative),*/
/*      checkpointInterval = $(checkpointInterval), seed = $(seed))*/
/*    val userDF = users.toDF("id", "features")*/
/*    val itemDF = items.toDF("id", "features")*/
/*    val model = new ALSModel(uid, $(rank), userDF, itemDF).setParent(this)*/
/*    copyValues(model)*/
/*  }*/
/**/
/*  override def transformSchema(schema: StructType): StructType = {*/
/*    validateAndTransformSchema(schema)*/
/*  }*/
/**/
/*  override def copy(extra: ParamMap): ALS = defaultCopy(extra)*/
/*}*/

/**
 * :: DeveloperApi ::
 * An implementation of ALS that supports generic ID types, specialized for Int and Long. This is
 * exposed as a developer API for users who do need other ID types. But it is not recommended
 * because it increases the shuffle size and memory requirement during training. For simplicity,
 * users and items must have the same type. The number of distinct users/items should be smaller
 * than 2 billion.
 */
/*@DeveloperApi*/

case class Rating[@specialized(Int, Long) ID](user: ID, item: ID, rating: Float)

object PNCG extends Logging {

  /**
   * :: DeveloperApi ::
   * Rating class for better code readability.
   */
  /*@DeveloperApi*/
  /*case class Rating[@specialized(Int, Long) ID](user: ID, item: ID, rating: Float)*/

  private type FactorRDD = RDD[(Int,FactorBlock)]
  private type FacTup = (FactorRDD,FactorRDD) // (user,items)

  private def logStdout(msg: String): Unit = {
		val time: Long = System.currentTimeMillis;
		logInfo(msg);
		println(time + ": " + msg);
	}

  /** Trait for least squares solvers applied to the normal equation. */
  private trait LeastSquaresNESolver extends Serializable {
    /** Solves a least squares problem with regularization (possibly with other constraints). */
    def solve(ne: NormalEquation, lambda: Double): Array[Float]
  }

  /** Cholesky solver for least square problems. */
  private class CholeskySolver extends LeastSquaresNESolver {

    private val upper = "U"

    /**
     * Solves a least squares problem with L2 regularization:
     *
     *   min norm(A x - b)^2^ + lambda * norm(x)^2^
     *
     * @param ne a [[NormalEquation]] instance that contains AtA, Atb, and n (number of instances)
     * @param lambda regularization constant
     * @return the solution x
     */
    override def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
      val k = ne.k
      // Add scaled lambda to the diagonals of AtA.
      var i = 0
      var j = 2
      while (i < ne.triK) {
        ne.ata(i) += lambda
        i += j
        j += 1
      }
      val info = new intW(0)
      lapack.dppsv(upper, k, 1, ne.ata, ne.atb, k, info)
      val code = info.`val`
      assert(code == 0, s"lapack.dppsv returned $code.")
      val x = new Array[Float](k)
      i = 0
      while (i < k) {
        x(i) = ne.atb(i).toFloat
        i += 1
      }
      ne.reset()
      x
    }
  }

  /** NNLS solver. */
  private class NNLSSolver extends LeastSquaresNESolver {
    private var rank: Int = -1
    private var workspace: NNLS.Workspace = _
    private var ata: Array[Double] = _
    private var initialized: Boolean = false

    private def initialize(rank: Int): Unit = {
      if (!initialized) {
        this.rank = rank
        workspace = NNLS.createWorkspace(rank)
        ata = new Array[Double](rank * rank)
        initialized = true
      } else {
        require(this.rank == rank)
      }
    }

    /**
     * Solves a nonnegative least squares problem with L2 regularizatin:
     *
     *   min_x_  norm(A x - b)^2^ + lambda * n * norm(x)^2^
     *   subject to x >= 0
     */
    override def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
      val rank = ne.k
      initialize(rank)
      fillAtA(ne.ata, lambda)
      val x = NNLS.solve(ata, ne.atb, workspace)
      ne.reset()
      x.map(x => x.toFloat)
    }

    /**
     * Given a triangular matrix in the order of fillXtX above, compute the full symmetric square
     * matrix that it represents, storing it into destMatrix.
     */
    private def fillAtA(triAtA: Array[Double], lambda: Double) {
      var i = 0
      var pos = 0
      var a = 0.0
      while (i < rank) {
        var j = 0
        while (j <= i) {
          a = triAtA(pos)
          ata(i * rank + j) = a
          ata(j * rank + i) = a
          pos += 1
          j += 1
        }
        ata(i * rank + i) += lambda
        i += 1
      }
    }
  }

  /**
   * Representing a normal equation to solve the following weighted least squares problem:
   *
   * minimize \sum,,i,, c,,i,, (a,,i,,^T^ x - b,,i,,)^2^ + lambda * x^T^ x.
   *
   * Its normal equation is given by
   *
   * \sum,,i,, c,,i,, (a,,i,, a,,i,,^T^ x - b,,i,, a,,i,,) + lambda * x = 0.
   */
  private class NormalEquation(val k: Int) extends Serializable {

    /** Number of entries in the upper triangular part of a k-by-k matrix. */
    val triK = k * (k + 1) / 2
    /** A^T^ * A */
    val ata = new Array[Double](triK)
    /** A^T^ * b */
    val atb = new Array[Double](k)

    private val da = new Array[Double](k)
    private val upper = "U"

    private def copyToDouble(a: Array[Float]): Unit = {
      var i = 0
      while (i < k) {
        da(i) = a(i)
        i += 1
      }
    }

    /** Adds an observation. */
    def add(a: Array[Float], b: Double, c: Double = 1.0): this.type = {
      require(c >= 0.0)
      require(a.length == k)
      copyToDouble(a)
      blas.dspr(upper, k, c, da, 1, ata)
      if (b != 0.0) {
        blas.daxpy(k, c * b, da, 1, atb, 1)
      }
      this
    }

    /** Merges another normal equation object. */
    def merge(other: NormalEquation): this.type = {
      require(other.k == k)
      blas.daxpy(ata.length, 1.0, other.ata, 1, ata, 1)
      blas.daxpy(atb.length, 1.0, other.atb, 1, atb, 1)
      this
    }

    /** Resets everything to zero, which should be called after each solve. */
    def reset(): Unit = {
      ju.Arrays.fill(ata, 0.0)
      ju.Arrays.fill(atb, 0.0)
    }
  }

  /**
   * Factor block that stores factors (Array[Float]) in an Array.
   */
  private type FactorBlock = Array[Array[Float]]

  /**
   * Out-link block that stores, for each dst (item/user) block, which src (user/item) factors to
   * send. For example, outLinkBlock(0) contains the local indices (not the original src IDs) of the
   * src factors in this block to send to dst block 0.
   */
  private type OutBlock = Array[Array[Int]]

  /**
   * In-link block for computing src (user/item) factors. This includes the original src IDs
   * of the elements within this block as well as encoded dst (item/user) indices and corresponding
   * ratings. The dst indices are in the form of (blockId, localIndex), which are not the original
   * dst IDs. To compute src factors, we expect receiving dst factors that match the dst indices.
   * For example, if we have an in-link record
   *
   * {srcId: 0, dstBlockId: 2, dstLocalIndex: 3, rating: 5.0},
   *
   * and assume that the dst factors are stored as dstFactors: Map[Int, Array[Array[Float]]], which
   * is a blockId to dst factors map, the corresponding dst factor of the record is dstFactor(2)(3).
   *
   * We use a CSC-like (compressed sparse column) format to store the in-link information. So we can
   * compute src factors one after another using only one normal equation instance.
   *
   * @param srcIds src ids (ordered)
   * @param dstPtrs dst pointers. Elements in range [dstPtrs(i), dstPtrs(i+1)) of dst indices and
   *                ratings are associated with srcIds(i).
   * @param dstEncodedIndices encoded dst indices
   * @param ratings ratings
   *
   * @see [[LocalIndexEncoder]]
   */
  private case class InBlock[@specialized(Int, Long) ID: ClassTag](
      srcIds: Array[ID],
      dstPtrs: Array[Int],
      dstEncodedIndices: Array[Int],
      ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = ratings.length
    require(dstEncodedIndices.length == size)
    require(dstPtrs.length == srcIds.length + 1)
  }

  /**
   * Initializes factors randomly given the in-link blocks.
   *
   * @param inBlocks in-link blocks
   * @param rank rank
   * @return initialized factor blocks
   */
  private def initialize[ID](
      inBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      seed: Long): RDD[(Int, FactorBlock)] = {
    // Choose a unit vector uniformly at random from the unit sphere, but from the
    // "first quadrant" where all elements are nonnegative. This can be done by choosing
    // elements distributed as Normal(0,1) and taking the absolute value, and then normalizing.
    // This appears to create factorizations that have a slightly better reconstruction
    // (<1%) compared picking elements uniformly at random in [0,1].
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new XORShiftRandom(byteswap64(seed ^ srcBlockId))
      val factors = Array.fill(inBlock.srcIds.length) {
        val factor = Array.fill(rank)(random.nextGaussian().toFloat)
        val nrm = blas.snrm2(rank, factor, 1)
        blas.sscal(rank, 1.0f / nrm, factor, 1)
        factor
      }
      (srcBlockId, factors)
    }
  }

  /**
   * A rating block that contains src IDs, dst IDs, and ratings, stored in primitive arrays.
   */
  private case class RatingBlock[@specialized(Int, Long) ID: ClassTag](
      srcIds: Array[ID],
      dstIds: Array[ID],
      ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = srcIds.length
    require(dstIds.length == srcIds.length)
    require(ratings.length == srcIds.length)
  }

  /**
   * Builder for [[RatingBlock]]. [[mutable.ArrayBuilder]] is used to avoid boxing/unboxing.
   */
  private class RatingBlockBuilder[@specialized(Int, Long) ID: ClassTag]
    extends Serializable {

    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstIds = mutable.ArrayBuilder.make[ID]
    private val ratings = mutable.ArrayBuilder.make[Float]
    var size = 0

    /** Adds a rating. */
    def add(r: Rating[ID]): this.type = {
      size += 1
      srcIds += r.user
      dstIds += r.item
      ratings += r.rating
      this
    }

    /** Merges another [[RatingBlockBuilder]]. */
    def merge(other: RatingBlock[ID]): this.type = {
      size += other.srcIds.length
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.ratings
      this
    }

    /** Builds a [[RatingBlock]]. */
    def build(): RatingBlock[ID] = {
      RatingBlock[ID](srcIds.result(), dstIds.result(), ratings.result())
    }
  }

  /**
   * Partitions raw ratings into blocks.
   *
   * @param ratings raw ratings
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   *
   * @return an RDD of rating blocks in the form of ((srcBlockId, dstBlockId), ratingBlock)
   */
  private def partitionRatings[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      srcPart: Partitioner,
      dstPart: Partitioner): RDD[((Int, Int), RatingBlock[ID])] = {

     /* The implementation produces the same result as the following but generates less objects.

     ratings.map { r =>
       ((srcPart.getPartition(r.user), dstPart.getPartition(r.item)), r)
     }.aggregateByKey(new RatingBlockBuilder)(
         seqOp = (b, r) => b.add(r),
         combOp = (b0, b1) => b0.merge(b1.build()))
       .mapValues(_.build())
     */

    val numPartitions = srcPart.numPartitions * dstPart.numPartitions
    ratings.mapPartitions { iter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =>
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId
        val builder = builders(idx)
        builder.add(r)
        if (builder.size >= 2048) { // 2048 * (3 * 4) = 24k
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =>
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName("ratingBlocks")
  }

  /**
   * Builder for uncompressed in-blocks of (srcId, dstEncodedIndex, rating) tuples.
   * @param encoder encoder for dst indices
   */
  private class UncompressedInBlockBuilder[@specialized(Int, Long) ID: ClassTag](
      encoder: LocalIndexEncoder)(
      implicit ord: Ordering[ID]) {

    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstEncodedIndices = mutable.ArrayBuilder.make[Int]
    private val ratings = mutable.ArrayBuilder.make[Float]

    /**
     * Adds a dst block of (srcId, dstLocalIndex, rating) tuples.
     *
     * @param dstBlockId dst block ID
     * @param srcIds original src IDs
     * @param dstLocalIndices dst local indices
     * @param ratings ratings
     */
    def add(
        dstBlockId: Int,
        srcIds: Array[ID],
        dstLocalIndices: Array[Int],
        ratings: Array[Float]): this.type = {
      val sz = srcIds.length
      require(dstLocalIndices.length == sz)
      require(ratings.length == sz)
      this.srcIds ++= srcIds
      this.ratings ++= ratings
      var j = 0
      while (j < sz) {
        this.dstEncodedIndices += encoder.encode(dstBlockId, dstLocalIndices(j))
        j += 1
      }
      this
    }

    /** Builds a [[UncompressedInBlock]]. */
    def build(): UncompressedInBlock[ID] = {
      new UncompressedInBlock(srcIds.result(), dstEncodedIndices.result(), ratings.result())
    }
  }

  /**
   * A block of (srcId, dstEncodedIndex, rating) tuples stored in primitive arrays.
   */
  private class UncompressedInBlock[@specialized(Int, Long) ID: ClassTag](
      val srcIds: Array[ID],
      val dstEncodedIndices: Array[Int],
      val ratings: Array[Float])(
      implicit ord: Ordering[ID]) {

    /** Size the of block. */
    def length: Int = srcIds.length

    /** Count the number of ratings per user/item 
     */
    def countRatings(): Array[Float] = {
      val len = length
      assert(len > 0, "Empty in-link block should not exist.")
      sort()
      val dstCountsBuilder = mutable.ArrayBuilder.make[Float]
      var preSrcId = srcIds(0)
      var curCount = 1
      var i = 1
      var j = 0
      while (i < len) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount

      dstCountsBuilder.result()
    }

    /**
     * Compresses the block into an [[InBlock]]. The algorithm is the same as converting a
     * sparse matrix from coordinate list (COO) format into compressed sparse column (CSC) format.
     * Sorting is done using Spark's built-in Timsort to avoid generating too many objects.
     */
    def compress(): InBlock[ID] = {
      val sz = length
      assert(sz > 0, "Empty in-link block should not exist.")
      sort()
      val uniqueSrcIdsBuilder = mutable.ArrayBuilder.make[ID]
      val dstCountsBuilder = mutable.ArrayBuilder.make[Int]
      var preSrcId = srcIds(0)
      uniqueSrcIdsBuilder += preSrcId
      var curCount = 1
      var i = 1
      var j = 0
      while (i < sz) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          uniqueSrcIdsBuilder += srcId
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount
      val uniqueSrcIds = uniqueSrcIdsBuilder.result()
      val numUniqueSrdIds = uniqueSrcIds.length
      val dstCounts = dstCountsBuilder.result()
      val dstPtrs = new Array[Int](numUniqueSrdIds + 1)
      var sum = 0
      i = 0
      while (i < numUniqueSrdIds) {
        sum += dstCounts(i)
        i += 1
        dstPtrs(i) = sum
      }
      InBlock(uniqueSrcIds, dstPtrs, dstEncodedIndices, ratings)
    }

    private def sort(): Unit = {
      val sz = length
      // Since there might be interleaved log messages, we insert a unique id for easy pairing.
      /*val sortId = Utils.random.nextInt()*/
      val sortId = (new Random()).nextInt()
      logDebug(s"Start sorting an uncompressed in-block of size $sz. (sortId = $sortId)")
      val start = System.nanoTime()
      val sorter = new Sorter(new UncompressedInBlockSort[ID])
      sorter.sort(this, 0, length, Ordering[KeyWrapper[ID]])
      val duration = (System.nanoTime() - start) / 1e9
      logDebug(s"Sorting took $duration seconds. (sortId = $sortId)")
    }
  }

  /**
   * A wrapper that holds a primitive key.
   *
   * @see [[UncompressedInBlockSort]]
   */
  private class KeyWrapper[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID]) extends Ordered[KeyWrapper[ID]] {

    var key: ID = _

    override def compare(that: KeyWrapper[ID]): Int = {
      ord.compare(key, that.key)
    }

    def setKey(key: ID): this.type = {
      this.key = key
      this
    }
  }

  /**
   * [[SortDataFormat]] of [[UncompressedInBlock]] used by [[Sorter]].
   */
  private class UncompressedInBlockSort[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID])
    extends SortDataFormat[KeyWrapper[ID], UncompressedInBlock[ID]] {

    override def newKey(): KeyWrapper[ID] = new KeyWrapper()

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int,
        reuse: KeyWrapper[ID]): KeyWrapper[ID] = {
      if (reuse == null) {
        new KeyWrapper().setKey(data.srcIds(pos))
      } else {
        reuse.setKey(data.srcIds(pos))
      }
    }

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int): KeyWrapper[ID] = {
      getKey(data, pos, null)
    }

    private def swapElements[@specialized(Int, Float) T](
        data: Array[T],
        pos0: Int,
        pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override def swap(data: UncompressedInBlock[ID], pos0: Int, pos1: Int): Unit = {
      swapElements(data.srcIds, pos0, pos1)
      swapElements(data.dstEncodedIndices, pos0, pos1)
      swapElements(data.ratings, pos0, pos1)
    }

    override def copyRange(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int,
        length: Int): Unit = {
      System.arraycopy(src.srcIds, srcPos, dst.srcIds, dstPos, length)
      System.arraycopy(src.dstEncodedIndices, srcPos, dst.dstEncodedIndices, dstPos, length)
      System.arraycopy(src.ratings, srcPos, dst.ratings, dstPos, length)
    }

    override def allocate(length: Int): UncompressedInBlock[ID] = {
      new UncompressedInBlock(
        new Array[ID](length), new Array[Int](length), new Array[Float](length))
    }

    override def copyElement(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int): Unit = {
      dst.srcIds(dstPos) = src.srcIds(srcPos)
      dst.dstEncodedIndices(dstPos) = src.dstEncodedIndices(srcPos)
      dst.ratings(dstPos) = src.ratings(srcPos)
    }
  }

  /**
   * Computes the Gramian matrix of user or item factors, which is only used in implicit preference.
   * Caching of the input factors is handled in [[ALS#train]].
   */
  private def computeYtY(factorBlocks: RDD[(Int, FactorBlock)], rank: Int): NormalEquation = {
    factorBlocks.values.aggregate(new NormalEquation(rank))(
      seqOp = (ne, factors) => {
        factors.foreach(ne.add(_, 0.0))
        ne
      },
      combOp = (ne1, ne2) => ne1.merge(ne2))
  }

  /**
   * Computes the Gramian matrix of user or item factors in a ray form for (x + \alpha * p)
   * which is only used in implicit preference.
   * Caching of the input factors is handled in [[ALS#train]].
   */
  private def computeYtYRay(
    xs: RDD[(Int, FactorBlock)],
    ps: RDD[(Int, FactorBlock)], 
    rank: Int): (Array[Float],Array[Float],Array[Float]) = 
  {

    val numel = rank * (rank + 1) / 2
    /*blas.dspr2(upper, k, c, da, 1, ata)*/
    xs.join(ps).values.aggregate((Array.ofDim[Float](numel), Array.ofDim[Float](numel), Array.ofDim[Float](numel)))(
      seqOp = (matrices, factors) => {
        val YtY = matrices._1
        val YtP = matrices._2
        val PtP = matrices._3
        val x = factors._1
        val p = factors._2
        val nfactors = x.length
        var k = 0
        while (k < nfactors) {
          //YtY
          blas.sspr("U", rank, 1f, x(k), 1, YtY); 
          //YtP + PtY
          blas.sspr2("U", rank, 1f, x(k), 1, p(k), 1, YtP); 
          // PtP
          blas.sspr("U", rank, 1f, p(k), 1, PtP); 
        }
        (YtY,YtP,PtP)
      },
      combOp = (matrices1, matrices2) => {
        // add two matrices together
        val YtY = matrices1._1
        val YtP = matrices1._2
        val PtP = matrices1._3
        blas.saxpy(numel,1f,matrices2._1,1,YtY,1)
        blas.saxpy(numel,1f,matrices2._2,1,YtP,1)
        blas.saxpy(numel,1f,matrices2._3,1,PtP,1)
        (YtY,YtP,PtP)
      }
    )
  }

  /**
   * Encoder for storing (blockId, localIndex) into a single integer.
   *
   * We use the leading bits (including the sign bit) to store the block id and the rest to store
   * the local index. This is based on the assumption that users/items are approximately evenly
   * partitioned. With this assumption, we should be able to encode two billion distinct values.
   *
   * @param numBlocks number of blocks
   */
  private class LocalIndexEncoder(numBlocks: Int) extends Serializable {

    require(numBlocks > 0, s"numBlocks must be positive but found $numBlocks.")

    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1 << numLocalIndexBits) - 1

    /** Encodes a (blockId, localIndex) into a single integer. */
    def encode(blockId: Int, localIndex: Int): Int = {
      require(blockId < numBlocks)
      require((localIndex & ~localIndexMask) == 0)
      (blockId << numLocalIndexBits) | localIndex
    }

    /** Gets the block id from an encoded index. */
    @inline
    def blockId(encoded: Int): Int = {
      encoded >>> numLocalIndexBits
    }

    /** Gets the local index from an encoded index. */
    @inline
    def localIndex(encoded: Int): Int = {
      encoded & localIndexMask
    }
  }

  /** 
   * Compute blas.SCALE; x = a*x, and return a new factor block 
   * @param x a vector represented as a FactorBlock
   * @param a scalar
   */
  private def blockSCAL(x: FactorBlock, a: Float): FactorBlock = 
  {
    val numVectors = x.length
    val rank = x(0).length
    val result: FactorBlock = x.map(_.clone())

    var k = 0;
    while (k < numVectors)
    {
      blas.sscal(rank,a,result(k),1)
      k += 1
    }
    result
  }

  /** 
   * compute a*x + b*y, and return a new factor block 
   * @param x a vector represented as a FactorBlock
   * @param y a vector represented as a FactorBlock
   * @param a a scalar
   */
  private def blockAXPBY(a: Float, x: FactorBlock, b: Float, y: FactorBlock): FactorBlock =
  {
    val numVectors = x.length
    val rank = x(0).length
    val result: FactorBlock = y.map(_.clone())

    var k = 0;
    while (k < numVectors)
    {
      //first scale b*y
      blas.sscal(rank,b,result(k),1)

      //y := a*x + (b*y)
      blas.saxpy(rank,a,x(k),1,result(k),1)
      k += 1
    }
    result
  }

  /** 
   * compute a*x + y, and return a new factor block 
   * @param a a scalar
   * @param x a vector represented as a FactorBlock
   * @param y a vector represented as a FactorBlock
   */
  private def blockAXPY(a: Float, x: FactorBlock, y: FactorBlock): FactorBlock =
  {
    val numVectors = x.length
    val rank = x(0).length
    val result: FactorBlock = y.map(_.clone())
    var k = 0
    while (k < numVectors)
    {
      blas.saxpy(rank,a,x(k),1,result(k),1)
      k += 1
    }
    result
  }

  /** 
   * compute x dot y, and return a scalar
   * @param x a vector represented as a FactorBlock
   * @param y a vector represented as a FactorBlock
   */
  private def blockDOT(x: FactorBlock, y: FactorBlock): Float = 
  {
    val numVectors = x.length
    val rank = x(0).length
    var result: Float = 0.0f
    var k = 0
    while (k < numVectors)
    {
      result += blas.sdot(rank,x(k),1,y(k),1)
      k += 1
    }
    result
  }

  /** 
   * compute x dot x, and return a scalar
   * @param x a vector represented as a FactorBlock
   */
  private def blockNRMSQR(x: FactorBlock): Float = 
  {
    val numVectors = x.length
    val rank = x(0).length
    var result: Float = 0.0f
    var norm: Float = 0.0f
    var k = 0
    while (k < numVectors)
    {
      norm = blas.snrm2(rank,x(k),1)
      result += norm * norm
      /*result += blas.sdot(rank,x(k),1,x(k),1)*/
      k += 1
    }
    result
  }

  /** 
   * compute dot product, and return a scalar
   * @param xs RDD of FactorBlocks
   * @param ys RDD of FactorBlocks
   */
  private def rddDOT(xs: FactorRDD, ys: FactorRDD): Float = {
    xs.join(ys)
      .map{case (_,(x,y)) => blockDOT(x,y)}
      .reduce{_+_}
  }


  /** 
   * compute x dot x, and return a scalar
   * @param xs RDD of FactorBlocks
   */
  private def rddNORMSQR(xs: FactorRDD): Float = {
    xs.map{case (_,x) => blockNRMSQR(x)}
      .reduce(_+_)
  }

  /** 
   * compute 2-norm of an RDD of FactorBlocks
   * @param xs RDD of FactorBlocks
   */
  private def rddNORM2(xs: FactorRDD): Float = {
    math.sqrt(rddNORMSQR(xs).toDouble).toFloat
  }

  /** 
   * compute a*x + b*y, for a FactorRDD = RDD[(Int, FactorBlock)]
   * @param x an RDD of vectors represented as a FactorBlock
   * @param y an RDD of vectors represented as a FactorBlock
   * @param a a scalar
   * @param b a scalar
   */
  private def rddAXPBY(a: Float, x: FactorRDD, b: Float, y: FactorRDD): FactorRDD = {
    x.join(y).mapValues{case(xblock,yblock) => blockAXPBY(a,xblock,b,yblock)}
  }

  /** 
   * compute a*x + y, for a FactorRDD = RDD[(Int, FactorBlock)]
   * @param x an RDD of vectors represented as a FactorBlock
   * @param y an RDD of vectors represented as a FactorBlock
   * @param a a scalar
   */
  private def rddAXPY(a: Float, x: FactorRDD, y: FactorRDD): FactorRDD = 
  {
    x.join(y).mapValues{case (xblock,yblock) => 
      blockAXPY(a,xblock,yblock)
    }
  }

  /**
   * Partitioner used by ALS. We requires that getPartition is a projection. That is, for any key k,
   * we have getPartition(getPartition(k)) = getPartition(k). Since the the default HashPartitioner
   * satisfies this requirement, we simply use a type alias here.
   */
  private type ALSPartitioner = org.apache.spark.HashPartitioner

  private class PolynomialMinimizer(funcCoeffs: Array[Float]) 
  {
    val degree = funcCoeffs.length - 1
    val gradCoeffs = funcCoeffs.zipWithIndex.tail.map{case (c,n) => n*c}
    val hessCoeffs = gradCoeffs.zipWithIndex.tail.map{case (c,n) => n*c}

    private def func(x: Float): Float = {
      var k = 1
      var sum = funcCoeffs(0)
      var pow = x
      while (k <= degree) {
        sum += funcCoeffs(k) * pow
        pow = pow * x
        k += 1
      }
      sum
    }
    private def grad(x: Float): Float = {
      var k = 1
      var sum = gradCoeffs(0)
      var pow = x
      while (k <= degree-1) {
        sum += gradCoeffs(k) * pow
        pow = pow * x
        k += 1
      }
      sum
    }
    private def hess(x: Float): Float = {
      var k = 1
      var sum = hessCoeffs(0)
      var pow = x
      while (k <= degree-2) {
        sum += hessCoeffs(k) * pow
        pow = pow * x
        k += 1
      }
      sum
    }
    // compute minimum around x0 using Newton's method
    // We find zeros of the gradient, since we are guaranteed that
    // the polynomial f(x) is decreasing at x = 0 for our problem
    def findMin(x0: Float, tol: Float, maxIters: Int): Float = 
    {
      var x = x0;
      var g = grad(x)
      var k = 1;
      /*logStdout(s"Rootfinder: $k: $x: $g: ${func(x)}")*/
      while ((math.abs(g) > tol) && (k <= maxIters)) {
        g = grad(x)
        x -= g / hess(x)
        k += 1
        /*logStdout(s"Rootfinder: $k: $x: $g: ${func(x)}")*/
      }
      x
    }

    def findMin(xs: Array[Float], tol: Float, maxIters: Int): Float = {
      val stepSizes = xs.map{x => findMin(x,tol,maxIters)}
      val minima = stepSizes.map{x => func(x)}
      val str = new StringBuilder(10 * xs.length)
      minima.foreach{x => str.append(x.toString + ",")}
      logStdout(s"findMin: found local minima: ${str.mkString}")
      str.clear
      stepSizes.foreach{x => str.append(x.toString + ",")}
      logStdout(s"findMin: stepsizes for minima: ${str.mkString}")
      stepSizes(minima.zipWithIndex.min._2)
    }
  }

  /**
   * :: DeveloperApi ::
   * Implementation of the nonlinearly-preconditioned CG accelerated ALS algorithm.
   */
  /*@DeveloperApi*/
  def train[ID: ClassTag]( // scalastyle:ignore
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1.0,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      nonnegative: Boolean = false,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval: Int = 10,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = 
  {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
    val blockRatings = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    val (userInBlocks, userOutBlocks, userCounts) =
      makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
    userCounts.count()
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks, itemCounts) =
      makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
    itemCounts.count()
    itemOutBlocks.count()

    val numUsers = computeDimension(userInBlocks);
    val numItems = computeDimension(itemInBlocks);
    val dof: Float = 1.0f * rank * (numUsers + numItems)
    logStdout(s"PNCG: Computing factors for $numUsers users and $numItems items: dof=$dof")

    var userCheckpointFile: Option[String] = None
    var itemCheckpointFile: Option[String] = None
    var direcUserCheckpointFile: Option[String] = None
    var direcItemCheckpointFile: Option[String] = None

    val shouldCheckpoint: Int => Boolean = (iter) =>
      (sc.getCheckpointDir.isDefined && (iter % checkpointInterval == 0))
      /*(iter % checkpointInterval == 0)*/

    def deleteCheckpointFile(file: Option[String]): Unit = {
      if (file != None){
        try {
          FileSystem.get(sc.hadoopConfiguration).delete(new Path(file.get), true)
        } catch {
          case e: IOException =>
            logWarning(s"Cannot delete checkpoint file $file:", e)
        }
      }
    }

    def preconditionItems(u: FactorRDD): FactorRDD = 
    {
      computeFactors(u, 
        userOutBlocks, 
        itemInBlocks, 
        rank, 
        regParam,
        userLocalIndexEncoder, 
        implicitPrefs,
        alpha,
        solver = solver)
    }

    def preconditionUsers(m: FactorRDD): FactorRDD = 
    {
      computeFactors(m, 
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder, 
        implicitPrefs,
        alpha,
        solver = solver)
    }

    def costFunc(x: FacTup): Float =
    {
      /*logStdout("costFunc: _init_");*/
      val usr = x._1
      val itm = x._2
      val sumSquaredErr: Float = evalFrobeniusCost(
        itm, 
        usr, 
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder
      )  
      /*logStdout("costFunc: var: sumSquaredErr: " + sumSquaredErr)*/
      val usrNorm: Float = evalTikhonovNorm(
        usr, 
        userCounts,
        rank,
        regParam
      ) 
      /*logStdout("costFunc: var: usrNorm: " + usrNorm)*/
      val itmNorm: Float = evalTikhonovNorm(
        itm, 
        itemCounts,
        rank,
        regParam
      )
      /*logStdout("costFunc: var: itmNorm: " + itmNorm)*/
      /*logStdout("costFunc: " + (sumSquaredErr + usrNorm + itmNorm))*/
      sumSquaredErr + usrNorm + itmNorm
    }

    def computeAlpha(
        userFac: FactorRDD,
        itemFac: FactorRDD,
        userDirec: FactorRDD,
        itemDirec: FactorRDD,
        userGrad: FactorRDD,
        itemGrad: FactorRDD,
        alpha0: Float,
        tol: Float,
        maxIters: Int,
        srcEncoder: LocalIndexEncoder
        ): Float = 
    {
      // form RDDs of (key,(x,p)) --- a "ray" with a point and a direction
      val userRay: RDD[(Int, (FactorBlock,FactorBlock))] = userFac.join(userDirec)
      val itemRay: RDD[(Int, (FactorBlock,FactorBlock))] = itemFac.join(itemDirec)

      val (xx_user,xp_user,pp_user) = evalTikhonovRayNorms(userRay,userCounts,rank,regParam)
      val (xx_item,xp_item,pp_item) = evalTikhonovRayNorms(itemRay,itemCounts,rank,regParam)

      /*val (userGrad,itemGrad) = computeGradient(userFac,itemFac,persist=false)*/

      val gradientProdDirec: Float = rddDOT(userGrad,userDirec) + rddDOT(itemGrad,itemDirec)

      type Ray = (FactorBlock,FactorBlock)

      def computeSquaredErrorFlat(
          block: InBlock[ID], 
          sortedRays: (Array[FactorBlock],Array[FactorBlock]),
          current: Ray): Array[Float] = 
      {
        val currentFactors = current._1
        val currentFactorDirecs = current._2
        val sortedSrcFactors = sortedRays._1
        val sortedSrcFactorDirecs = sortedRays._2
        val len = block.srcIds.length
        var j = 0

        val coeff: Array[Float] = Array.ofDim(5)

        while (j < len) 
        {
          val y = currentFactors(j)
          val q = currentFactorDirecs(j)
          var i = block.dstPtrs(j)

          while (i < block.dstPtrs(j + 1)) {
            val encoded = block.dstEncodedIndices(i)
            val blockId = srcEncoder.blockId(encoded)
            val localIndex = srcEncoder.localIndex(encoded)

            val x = sortedSrcFactors(blockId)(localIndex)
            val p = sortedSrcFactorDirecs(blockId)(localIndex)

            // compute the necessary dot products
            val xy = blas.sdot(rank,x,1,y,1)
            val xq = blas.sdot(rank,x,1,q,1)
            val py = blas.sdot(rank,p,1,y,1)
            val pq = blas.sdot(rank,p,1,q,1)

            // avoid catastrophic cancellation where possible:
            // don't compute (xy - r) in coeff(0) or coeff(2)
            if (implicitPrefs) {
              val r = if (block.ratings(i) == 0) 0f else 1f
              val c = (1 + alpha * block.ratings(i)).toFloat
              coeff(0) += c*( (xy*xy  - 2*xy*r) + r*r )
              coeff(1) += 2*c*(xq + py)*(xy - r)
              coeff(2) += c*( 2*pq*xy + xq*xq + py*(py + 2*xq) - 2*r*pq )
              coeff(3) += 2*c*pq*(xq + py)
              coeff(4) += c*pq*pq
            } else {
              val r = block.ratings(i)
              coeff(0) += (xy*xy  - 2*xy*r) + r*r
              coeff(1) += 2*(xq + py)*(xy - r)
              coeff(2) += 2*pq*xy + xq*xq + py*(py + 2*xq) - 2*r*pq
              coeff(3) += 2*pq*(xq + py)
              coeff(4) += pq*pq
            }

            i += 1
          }
          j += 1
        }
        coeff
      }

      val coeff: Array[Float] = 
        makeFrobeniusCostRDD(
          itemFac, 
          itemDirec,
          userFac, 
          userDirec,
          itemOutBlocks, 
          userInBlocks, 
          rank
        )
        .map{case ( (block,rays),ray) => computeSquaredErrorFlat(block,rays,ray)}
        .reduce{ (x,y) => 
          val p = y.clone
          blas.saxpy(5,1.0f,x,1,p,1)
          p
        }
      // add the tikhonov regularization to the coefficients
      coeff(0) += xx_user + xx_item
      coeff(1) += 2*(xp_user + xp_item)
      coeff(2) += 2*(pp_user + pp_item)

      // this coefficient doesn't actually matter; easier to read if set to zero
      coeff(0) = 0
      val polyMin = new PolynomialMinimizer(coeff)
      // find the best minimum near both alpha0 and at 20, which is far;
      // typical values for alpha are in [0,2]
      polyMin.findMin(Array(alpha0,20f), tol, maxIters)
    }

    val seedGen = new XORShiftRandom(seed)
    var users = initialize(userInBlocks, rank, seedGen.nextLong()).cache
    var items = initialize(itemInBlocks, rank, seedGen.nextLong()).cache

    /*logStdout("PNCG: 0: " + (users.count + items.count))*/

    var users_pc: FactorRDD = preconditionUsers(items).cache()
    var items_pc: FactorRDD = preconditionItems(users_pc)

    // compute preconditioned gradients; g = x - x_pc
    var gradUser_pc: FactorRDD = rddAXPY(-1.0f,users_pc,users).cache()
    var gradItem_pc: FactorRDD = rddAXPY(-1.0f,items_pc,items).cache()

    // initialize variables for the previous iteration's gradients
    var gradUser_pc_old: FactorRDD = gradUser_pc
    var gradItem_pc_old: FactorRDD = gradItem_pc

    // initial search direction to -gradient_preconditioned 
    var direcUser: FactorRDD = gradUser_pc.mapValues{x => blockSCAL(x,-1.0f)}.cache()
    var direcItem: FactorRDD = gradItem_pc.mapValues{x => blockSCAL(x,-1.0f)}.cache()

    var gradUser: FactorRDD = evalGradient(items,users,itemOutBlocks,userInBlocks,rank,regParam,itemLocalIndexEncoder).cache()
    var gradItem: FactorRDD = evalGradient(users,items,userOutBlocks,itemInBlocks,rank,regParam,userLocalIndexEncoder).cache()

    var gradUser_old: FactorRDD = gradUser
    var gradItem_old: FactorRDD = gradItem

    // compute g^T * g
    var gradTgrad = rddDOT(gradUser,gradUser_pc) + rddDOT(gradItem,gradItem_pc);
    var gradTgrad_old = gradTgrad;

    val restartTol: Float = 0.1f
    val alpha0: Float = 0.0f
    var beta_pncg: Float = gradTgrad
    var alpha_pncg: Float = alpha0

    logStdout(s"PNCG: 0: $alpha_pncg: $beta_pncg: ${1/dof*math.sqrt(rddNORMSQR(gradUser)+rddNORMSQR(gradItem))}: ${costFunc((users,items))}")
    for (iter <- 1 to maxIter) 
    {
      alpha_pncg = computeAlpha(users,items,direcUser,direcItem,gradUser,gradItem,
        alpha0,
        1e-8f,
        10,
        itemLocalIndexEncoder
      )

      // x_{k+1} = x_k + \alpha * p_k
      users = rddAXPY(alpha_pncg, direcUser, users).cache()
      items = rddAXPY(alpha_pncg, direcItem, items).cache()
      /*logStdout(s"PNCG: updated users with ${users.count} partitions")*/
      /*logStdout(s"PNCG: updated items with ${items.count} partitions")*/

      /*if (sc.checkpointDir.isDefined && (iter % checkpointInterval == 0))*/
      if (iter % checkpointInterval == 0)
      {
        logStdout(s"PNCG: Checkpointing users/items at iter $iter")
        users.checkpoint()
        items.checkpoint()
        items.count()
        users.count()
        deleteCheckpointFile(userCheckpointFile)
        deleteCheckpointFile(itemCheckpointFile)
        userCheckpointFile = users.getCheckpointFile
        itemCheckpointFile = items.getCheckpointFile
      }

      // precondition x with ALS
      // \bar{x} = P * \x_{k+1}
      users_pc = preconditionUsers(items).cache
      items_pc = preconditionItems(users_pc)

      // compute the preconditioned gradient
      // g = x_{k+1} - \bar{x} 
      gradUser_pc = rddAXPY(-1.0f,users_pc,users).cache // x - x_pc
      gradItem_pc = rddAXPY(-1.0f,items_pc,items).cache // x - x_pc

      // PR
      gradUser = evalGradient(items,users,itemOutBlocks,userInBlocks,rank,regParam,itemLocalIndexEncoder,implicitPrefs,alpha).cache()
      gradItem = evalGradient(users,items,userOutBlocks,itemInBlocks,rank,regParam,userLocalIndexEncoder,implicitPrefs,alpha).cache()

      gradTgrad = rddDOT(gradUser,gradUser_pc) + rddDOT(gradItem,gradItem_pc);

      //original beta_pncg version:
      /*beta_pncg = (gradTgrad - (rddDOT(gradUser,gradUser_pc_old) + rddDOT(gradItem,gradItem_pc_old)) ) / gradTgrad_old*/

      beta_pncg = (gradTgrad - (rddDOT(gradUser_old,gradUser_pc) + rddDOT(gradItem_old,gradItem_pc)) ) / gradTgrad_old

      // compute the restart condition from Nocedal & Wright, Numerical Optimization 2006
      /*val shouldRestart: Boolean = {*/
      /*  val projection = rddDOT(gradUser_pc, gradUser_pc_old) + rddDOT(gradItem_pc, gradItem_pc_old)*/
      /*  val scaling = rddNORMSQR(gradUser_pc) + rddNORMSQR(gradItem_pc)*/
      /*  val restartTest = projection / scaling*/
      /*  val result = restartTest > restartTol*/
      /*  if (result) */
      /*    logStdout(s"PNCG: $iter: restartTest = $restartTest > $restartTol: Restarting with steepest descent")*/
      /*  result*/
      /*}*/
      /*if (beta_pncg < 0) */
      /*  logStdout(s"PNCG: $iter: beta < 0: Restarting with steepest descent")*/
      /*else if (shouldRestart)*/
      
      // p_{k+1} = -g + \beta * p_k
      direcUser = {
        if (beta_pncg < 0) {
          logStdout(s"PNCG: beta < 0 in iter $iter: Restarting with steepest descent")
          gradUser_pc.mapValues{x => blockSCAL(x,-1.0f)}
        } else {
          rddAXPBY(-1.0f,gradUser_pc,beta_pncg,direcUser)
        }
      }.cache
      direcItem = {
        if (beta_pncg < 0) {
          gradItem_pc.mapValues{x => blockSCAL(x,-1.0f)}
        } else {
          rddAXPBY(-1.0f,gradItem_pc,beta_pncg,direcItem)
        }
      }.cache

      /*if (sc.checkpointDir.isDefined && (iter % checkpointInterval == 0))*/
      if (iter % checkpointInterval == 0)
      {
        logStdout(s"PNCG: Checkpointing users/item directionss at iter $iter")
        direcUser.checkpoint()
        direcItem.checkpoint()
        direcUser.count()
        direcItem.count()
        deleteCheckpointFile(direcUserCheckpointFile)
        deleteCheckpointFile(direcItemCheckpointFile)
        direcUserCheckpointFile = users.getCheckpointFile
        direcItemCheckpointFile = items.getCheckpointFile
      }

      // store old preconditioned gradient vectors for computing \beta
      gradTgrad_old = gradTgrad
      gradUser_pc_old = gradUser_pc
      gradItem_pc_old = gradItem_pc

      // store old gradient for computing the other version of \beta
      gradUser_old = gradUser
      gradItem_old = gradItem

      /*gradUser_pc_old = gradUser_pc.cache*/
      /*gradItem_pc_old = gradItem_pc.cache*/
      /*gradUser_pc_old.count*/
      /*gradItem_pc_old.count*/

      logStdout(s"PNCG: $iter: $alpha_pncg: $beta_pncg: ${1/dof * math.sqrt(rddNORMSQR(gradUser)+rddNORMSQR(gradItem))}: ${costFunc((users,items))}")
    }
    
    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(users)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and users.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(items)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      items.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

  /**
   * Creates in-blocks and out-blocks from rating blocks.
   * @param prefix prefix for in/out-block names
   * @param ratingBlocks rating blocks
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   * @return (in-blocks, out-blocks)
   */
  private def makeBlocks[ID: ClassTag](
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock[ID])],
      srcPart: Partitioner,
      dstPart: Partitioner,
      storageLevel: StorageLevel)(
      implicit srcOrd: Ordering[ID]): (RDD[(Int, InBlock[ID])], RDD[(Int, OutBlock)], RDD[(Int, Array[Float])]) = {

    /**
     * compute the local destination indices for each index i as
     * i_local = mod(i,N), where N is the nu
     */ 
    def computeLocalIndices(dstIds: Array[ID]): Array[Int] = {

      val start = System.nanoTime()
      val dstIdSet = new OpenHashSet[ID](1 << 20)
      dstIds.foreach(dstIdSet.add)

      // The implementation is a faster version of
      // val dstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
      val sortedDstIds = new Array[ID](dstIdSet.size)
      var i = 0
      var pos = dstIdSet.nextPos(0)
      while (pos != -1) {
        sortedDstIds(i) = dstIdSet.getValue(pos)
        pos = dstIdSet.nextPos(pos + 1)
        i += 1
      }
      assert(i == dstIdSet.size)
      Sorting.quickSort(sortedDstIds)
      val len = sortedDstIds.length
      val dstIdToLocalIndex = new OpenHashMap[ID, Int](len)

      i = 0
      while (i < len) {
        dstIdToLocalIndex.update(sortedDstIds(i), i)
        i += 1
      }
      logDebug("Converting to local indices took " 
        + (System.nanoTime() - start) / 1e9 
        + " seconds.")

      dstIds.map(dstIdToLocalIndex.apply)
    }

    type UncompressedCols = (Int, Array[ID], Array[Int], Array[Float])

    def toUncompressedCols(key: (Int,Int), block: RatingBlock[ID]): (Int, UncompressedCols) = {
      val localBlockInds: Array[Int] = computeLocalIndices(block.dstIds)
      (key._1, (key._2, block.srcIds, localBlockInds, block.ratings) ) 
    }

    def toUncompressedSparseCols(iter: Iterable[UncompressedCols]): UncompressedInBlock[ID] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build()
    }

    def toCounts(iter: Iterable[UncompressedCols]): Array[Float] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build().countRatings()
    }

    def toCompressedSparseCols(iter: Iterable[UncompressedCols]): InBlock[ID] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build().compress()
    }

    def toOutLinkArray(block: InBlock[ID]): OutBlock = { 
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val activeIds = Array.fill(dstPart.numPartitions)(mutable.ArrayBuilder.make[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      while (i < block.srcIds.length) {
        var j = block.dstPtrs(i)
        ju.Arrays.fill(seen, false)
        while (j < block.dstPtrs(i + 1)) {
          val dstBlockId = encoder.blockId(block.dstEncodedIndices(j))
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i // add the local index in this out-block
            seen(dstBlockId) = true
          }
          j += 1
        }
        i += 1
      }
      activeIds.map { x =>
        x.result()
      }
    }

    val counts: RDD[(Int, Array[Float])] = ratingBlocks
      .map{ case(key,block) => toUncompressedCols(key,block) } //(BlockId, (Int, Array[ID], Array[Int], Array[Float]) )
      .groupByKey(new ALSPartitioner(srcPart.numPartitions))
      .mapValues(toCounts)
      .setName(prefix + "RatingsCounts")
      .persist(storageLevel)

    val inBlocks = ratingBlocks
      .map{ case(key,block) => toUncompressedCols(key,block) } //(BlockId, (Int, Array[ID], Array[Int], Array[Float]) )
      .groupByKey(new ALSPartitioner(srcPart.numPartitions))
      .mapValues(toCompressedSparseCols)
      .setName(prefix + "InBlocks")
      .persist(storageLevel)

    val outBlocks = inBlocks
      .mapValues(toOutLinkArray)
      .setName(prefix + "OutBlocks")
      .persist(storageLevel)

    (inBlocks, outBlocks, counts)
  }

  /**
   * Evaluate the gradient function f(U,M), as in \cite{zhou2008largescale}
   *
   * Comments are written assuming the gradient WRT users is being calculated.
   * For, e.g. the ith user u_i:
   *  1/2 * df(u_i)/du_i = (M_i * M^T_i + \lambda * n_{u_i} )*u_i - M_{u_i}*R^T_{u_i}
   *
   * @param srcFactorBlocks src factors; the item factors, m_i
   * @param currentFactorBlocks current user factors, u_i
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return grad gradient vector 
   */
  private def evalGradient[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      currentFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0
      ): RDD[(Int, FactorBlock)] = 
  {
    val numSrcBlocks = srcFactorBlocks.partitions.length

    val YtY: Array[Float] = 
      if (implicitPrefs) 
        Some(computeYtY(srcFactorBlocks, rank)).get.ata.map(_.toFloat)
      else 
        Array()

    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = 
    {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def computeGradientBlock(
        block: InBlock[ID],  //{m_j}
        factorList: Iterable[(Int,FactorBlock)],
        current: FactorBlock 
        ): FactorBlock = 
    {
      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length

      // initialize array of gradient vectors
      val grad: Array[Array[Float]] = Array.fill(len)(Array.fill[Float](rank)(0f))

      var i = 0

      // loop over all users {u_i}
      while (i < len) 
      {
        // loop over all input {m_j} in block
        var j = block.dstPtrs(i)
        var num_factors = 0
        while (j < block.dstPtrs(i + 1)) 
        {
          val encoded = block.dstEncodedIndices(j)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex) //m_
          val rating = block.ratings(j)
          // scale the src factor by a
          val a: Float = { 
            if (implicitPrefs) {
              val p = if (rating > 0) 1f else 0f
              val c = (1 + alpha * math.abs(rating)).toFloat
              2*c*(blas.sdot(rank,current(i),1,srcFactor,1) - p)
            } else {
              2*(blas.sdot(rank,current(i),1,srcFactor,1) - rating)
            }
          }
          // y := a*x + y 
          blas.saxpy(rank,a,srcFactor,1,grad(i),1)
          j += 1
          num_factors += 1
        }
        // add \lambda * n * u_i
        val penaltyCoeff = {
          if (implicitPrefs) 
            regParam.toFloat 
          else 
            regParam.toFloat*num_factors
        }
        blas.saxpy(rank,2*penaltyCoeff,current(i),1,grad(i),1)

        // finally, if implicit, add the term 2*YtY*x_u to the gradient
        if (implicitPrefs) {
          /*logStdout(s"Multiplyting x with length ${current(i).length} by YtY with length ${YtY.length}");*/
          blas.sspmv("U",rank,2f,YtY,current(i),1,1f,grad(i),1)
        }
        i += 1
      }
      grad
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = 
      srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val gradient: RDD[(Int, FactorBlock)] = dstInBlocks
      .join(srcOut)
      .join(currentFactorBlocks)
      .mapValues{case ((block,factorTuple),fac) => computeGradientBlock(block,factorTuple,fac)}
      
      /*.cogroup(srcOut,currentFactorBlocks)*/
      //use .head since cogroup has produced Iterables
      /*.mapValues{case (block,factorTuple,fac) => */
      /*  computeGradientBlock(block.head,factorTuple.head,fac.head)*/
      /*}*/
    gradient
  }

  private def evalTikhonovRayNorms(
      ray: RDD[(Int, (FactorBlock,FactorBlock))],
      counts: RDD[(Int, Array[Float])],
      rank: Int,
      lambda: Double,
      implicitPrefs: Boolean = false): (Float,Float,Float) = 
  {
    def evalBlockNorms(x: FactorBlock, p: FactorBlock): (Array[Float],Array[Float],Array[Float]) = 
    {
      val numFactors: Int = x.length
      val xTx: Array[Float] = new Array[Float](numFactors)
      val xTp: Array[Float] = new Array[Float](numFactors)
      val pTp: Array[Float] = new Array[Float](numFactors)
      var j: Int = 0
      while (j < numFactors) {
        xTx(j) = blas.sdot(rank,x(j),1,x(j),1)
        xTp(j) = blas.sdot(rank,x(j),1,p(j),1)
        pTp(j) = blas.sdot(rank,p(j),1,p(j),1)
        j += 1
      }
      (xTx,xTp,pTp)
    }
    def evalFlatBlockNorms(x: FactorBlock, p: FactorBlock): (Float,Float,Float) = 
    {
      val numFactors: Int = x.length
      var xTx: Float = 0
      var xTp: Float = 0
      var pTp: Float = 0
      var j: Int = 0
      while (j < numFactors) {
        xTx += blas.sdot(rank,x(j),1,x(j),1)
        xTp += blas.sdot(rank,x(j),1,p(j),1)
        pTp += blas.sdot(rank,p(j),1,p(j),1)
        j += 1
      }
      (xTx,xTp,pTp)
    }
    def scaleByNumRatings(factorNorms: Array[Float], numRatings: Array[Float]): Float =
    {
      val numFactors: Int = factorNorms.length
      blas.sdot(numFactors,factorNorms,1,numRatings,1)
    }

    val factorNorms: (Float,Float,Float) = { 
      if (implicitPrefs) {
        ray
          .map{ case (_,(x,p)) => evalFlatBlockNorms(x,p) }
          .reduce{ (x,y) => (x._1+y._1, x._2+y._2, x._3 + y._3) }
      } else {
        ray
          .mapValues{ case(x,p) => evalBlockNorms(x,p) }
          .join(counts)
          .map{case (key,((xx,xp,pp),n)) => (scaleByNumRatings(xx,n),scaleByNumRatings(xp,n),scaleByNumRatings(pp,n)) }
          .reduce{ (x,y) => (x._1+y._1, x._2+y._2, x._3 + y._3) }
      }
    }

    val lam = lambda.toFloat
    (factorNorms._1 * lam, factorNorms._2 * lam, factorNorms._3 * lam)
  }

  /**
   * Evaluate the Tikhonov normalization for f(U,M)
   *
   * @param factors Array of Array[Float] factors; the item factors, m_i
   * @param counts the number of ratings associated with each factor, Array[Int] 
   * @param rank the size of a single factor vector
   *
   */
  private def evalTikhonovNorm(
      factors: RDD[(Int, FactorBlock)],
      counts: RDD[(Int, Array[Float])],
      rank: Int,
      lambda: Double
      ): Float = 
  {
    def evalBlockNorms(block: FactorBlock): Array[Float] = 
    {
      val numFactors: Int = block.length
      val result: Array[Float] = new Array[Float](numFactors)
      var j: Int = 0
      while (j < numFactors) {
        result(j) = blas.sdot(rank,block(j),1,block(j),1)
        j += 1
      }
      result
    }
    def scaleByNumRatings(factorNorms: Array[Float], numRatings: Array[Float]): Float =
    {
      val numFactors: Int = factorNorms.length
      var result: Float = 0f
      var j: Int = 0
      blas.sdot(numFactors,factorNorms,1,numRatings,1)
    }

    val factorNorm: Float = factors
      .mapValues{evalBlockNorms}
      .join(counts)
      .map{case (key,(f,n)) => scaleByNumRatings(f,n)}
      .reduce(_ + _)

    lambda.toFloat * factorNorm
  }

  /*
      val sumSquaredErr: Float = evalFrobeniusCost(
        itm, 
        itm_p, ***
        usr, 
        usr_p, ***
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder
      )  
   */

  private def makeFrobeniusCostRDD[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      srcFactorBlocksDirec: RDD[(Int, FactorBlock)],
      currentFactorBlocks: RDD[(Int, FactorBlock)],
      currentFactorBlocksDirec: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int) = 
  {
    type Ray = (FactorBlock,FactorBlock)

    val numSrcBlocks = srcFactorBlocks.partitions.length

    /*type BlockFacTuple = (OutBlock,FactorBlock)*/
    def filterFactorsToSend (
        srcBlockId: Int, 
        tup: (OutBlock, (FactorBlock, FactorBlock) )
        ) = 
    {
      val block = tup._1
      val x_factors = tup._2._1
      val p_factors = tup._2._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, (activeIndices.map(x_factors(_)), activeIndices.map(p_factors(_))) ))
        }
    }

    // Rays are: RDD[(key, (x, p) )]
    val usrRay: RDD[(Int,Ray)] = srcFactorBlocks.join(srcFactorBlocksDirec)

    val itmRay: RDD[(Int,Ray)] = currentFactorBlocks.join(currentFactorBlocksDirec)

    val srcOut: RDD[(Int, (Array[FactorBlock],Array[FactorBlock]) )] = srcOutBlocks
      .join(usrRay)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))
      .mapValues { iter =>
        val sortedFactor = new Array[FactorBlock](numSrcBlocks)
        val sortedFactorDirec = new Array[FactorBlock](numSrcBlocks)
        iter.foreach{ case (srcBlockId, ray) =>
            sortedFactor(srcBlockId) = ray._1
            sortedFactorDirec(srcBlockId) = ray._2
        }
        (sortedFactor,sortedFactorDirec)
      }

    val result: RDD[( (InBlock[ID], (Array[FactorBlock],Array[FactorBlock])),Ray)] = dstInBlocks
      .join(srcOut)
      .join(itmRay)
      .values

    result
  }

  /**
   * Compute the Frobenius norm part of the cost function for the current set of factors 
   *
   * @param srcFactorBlocks src factors
   * @param currentFactorBlocks current user factors, u_i
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return dst factors
   */
  private def evalFrobeniusCost[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      currentFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0
      ): Float = 
    {

    val numSrcBlocks = srcFactorBlocks.partitions.length
    val YtY = 
      if (implicitPrefs) 
        Some(computeYtY(srcFactorBlocks, rank)) 
      else 
        None

    /*type BlockFacTuple = (OutBlock,FactorBlock)*/
    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def computeSquaredError(
        block: InBlock[ID], 
        /*factorList: Iterable[(Int,FactorBlock)],*/
        factorList: Iterable[(Int,FactorBlock)],
        current: FactorBlock 
        ): Float = 
    {

      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length
      var j = 0
      var sumErrs: Double = 0
      while (j < len) 
      {
        var i = block.dstPtrs(j)
        while (i < block.dstPtrs(j + 1)) {
          val encoded = block.dstEncodedIndices(i)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex)
          val rating = block.ratings(i)
          val diff = blas.sdot(rank,current(j),1,srcFactor,1) - rating
          sumErrs += diff * diff
          i += 1
        }
        j += 1
      }
      sumErrs.toFloat
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val result: Double = dstInBlocks
      .join(srcOut)
      .join(currentFactorBlocks)
      /*.map{(key,((block,factorTuple),fac) ) => computeSquaredError(block,factorTuple,fac)}*/
      /*.cogroup(srcOut,currentFactorBlocks)*/
      .mapValues{case ( (block,factorTuple),fac) => 
        computeSquaredError(block,factorTuple,fac)
      }
      .values
      .reduce(_+_)

    result.toFloat
  }


  /**
   * Compute dst factors by constructing and solving least square problems.
   *
   * @param srcFactorBlocks src factors
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return dst factors
   */
  private def computeFactors[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      solver: LeastSquaresNESolver): RDD[(Int, FactorBlock)] = 
    {

    val numSrcBlocks = srcFactorBlocks.partitions.length
    val YtY = 
      if (implicitPrefs) 
        Some(computeYtY(srcFactorBlocks, rank)) 
      else 
        None

    /*type BlockFacTuple = (OutBlock,FactorBlock)*/
    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def solveNormalEqn(
        block: InBlock[ID], 
        factorList: Iterable[(Int,FactorBlock)] ): FactorBlock = {

      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length
      val dstFactors = new Array[Array[Float]](len)
      var j = 0
      val normEqn = new NormalEquation(rank)
      while (j < len) {
        normEqn.reset()
        if (implicitPrefs) {
          normEqn.merge(YtY.get)
        }
        var numExplicits = 0
        var i = block.dstPtrs(j)
        while (i < block.dstPtrs(j + 1)) {
          val encoded = block.dstEncodedIndices(i)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex)
          val rating = block.ratings(i)
          if (implicitPrefs) {
            if (rating > 0) {
              val c1 = alpha * math.abs(rating)
              normEqn.add(srcFactor, (c1 + 1.0) / c1, c1)
              numExplicits += 1
            }
          } else {
            normEqn.add(srcFactor, rating)
            numExplicits += 1
          }
          i += 1
        }
        dstFactors(j) = solver.solve(normEqn, numExplicits * regParam)
        j += 1
      }
      dstFactors
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = 
      srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val newFactors: RDD[(Int, FactorBlock)] = 
      dstInBlocks
      .join(srcOut)
      .mapValues{case (block, factorTuple) => solveNormalEqn(block,factorTuple)}

    newFactors
  }

  /**
   * :: DeveloperApi ::
   * Original Implementation of the ALS algorithm.
   */
  def trainALS[ID: ClassTag]( // scalastyle:ignore
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1.0,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      nonnegative: Boolean = false,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval: Int = 10,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = 
  {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
    val blockRatings = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    val (userInBlocks, userOutBlocks, userCounts) =
      makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks, itemCounts) =
      makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
    itemOutBlocks.count()
    val numUsers = computeDimension(userInBlocks);
    val numItems = computeDimension(itemInBlocks);
    /*val dof: Float = 1.0f * rank * (numUsers + numItems)*/
    val dof: Float = 1.0f;
    logStdout(s"PNCG: Computing factors for $numUsers users and $numItems items: dof=$dof")

    def costFunc(x: FacTup): Float =
    {
      /*logStdout("costFunc: _init_");*/
      val usr = x._1
      val itm = x._2
      val sumSquaredErr: Float = evalFrobeniusCost(
        itm, 
        usr, 
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder
      )  
      /*logStdout("costFunc: var: sumSquaredErr: " + sumSquaredErr)*/
      val usrNorm: Float = evalTikhonovNorm(
        usr, 
        userCounts,
        rank,
        regParam
      ) 
      /*logStdout("costFunc: var: usrNorm: " + usrNorm)*/
      val itmNorm: Float = evalTikhonovNorm(
        itm, 
        itemCounts,
        rank,
        regParam
      )
      /*logStdout("costFunc: var: itmNorm: " + itmNorm)*/
      /*logStdout("costFunc: " + (sumSquaredErr + usrNorm + itmNorm))*/
      sumSquaredErr + usrNorm + itmNorm
    }

    val seedGen = new XORShiftRandom(seed)
    var userFactors = initialize(userInBlocks, rank, seedGen.nextLong())
    var itemFactors = initialize(itemInBlocks, rank, seedGen.nextLong())
    var previousCheckpointFile: Option[String] = None
    val shouldCheckpoint: Int => Boolean = (iter) =>
      (sc.getCheckpointDir.isDefined && (iter % checkpointInterval == 0))

    val deletePreviousCheckpointFile: () => Unit = () =>
      previousCheckpointFile.foreach { file =>
        try {
          FileSystem.get(sc.hadoopConfiguration).delete(new Path(file), true)
        } catch {
          case e: IOException =>
            logWarning(s"Cannot delete checkpoint file $file:", e)
        }
      }

    var gradItem = evalGradient(userFactors,itemFactors,userOutBlocks,itemInBlocks,rank,regParam,userLocalIndexEncoder,implicitPrefs,alpha)
    var gradUser = evalGradient(itemFactors,userFactors,itemOutBlocks,userInBlocks,rank,regParam,itemLocalIndexEncoder,implicitPrefs,alpha)
    logStdout(s"ALS: 0: ${1/dof * math.sqrt(rddNORMSQR(gradUser) + rddNORMSQR(gradItem))}: ${costFunc((userFactors,itemFactors))}")
    if (implicitPrefs) {
      for (iter <- 1 to maxIter) {
        userFactors.setName(s"userFactors-$iter").persist(intermediateRDDStorageLevel)
        val previousItemFactors = itemFactors
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, implicitPrefs, alpha, solver)
        previousItemFactors.unpersist()
        itemFactors.setName(s"itemFactors-$iter").persist(intermediateRDDStorageLevel)
        // TODO: Generalize PeriodicGraphCheckpointer and use it here.
        if (shouldCheckpoint(iter)) {
          itemFactors.checkpoint() // itemFactors gets materialized in computeFactors.
        }
        val previousUserFactors = userFactors
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, implicitPrefs, alpha, solver)
        if (shouldCheckpoint(iter)) {
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        previousUserFactors.unpersist()

        gradItem = evalGradient(userFactors,itemFactors,userOutBlocks,itemInBlocks,rank,regParam,userLocalIndexEncoder,implicitPrefs,alpha)
        logStdout(s"ALS: $iter: ${1/dof * math.sqrt(rddNORMSQR(gradItem))}: ${costFunc((userFactors,itemFactors))}")
      }
    } else {
      for (iter <- 1 until maxIter) {
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, solver = solver).cache
        if (shouldCheckpoint(iter)) {
          itemFactors.checkpoint()
          itemFactors.count() // checkpoint item factors and cut lineage
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, solver = solver).cache

        gradItem = evalGradient(userFactors,itemFactors,userOutBlocks,itemInBlocks,rank,regParam,userLocalIndexEncoder,implicitPrefs,alpha)
        logStdout(s"ALS: $iter: ${1/dof * math.sqrt(rddNORMSQR(gradItem))}: ${costFunc((userFactors,itemFactors))}")
      }
    }
    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }
  private def computeDimension[ID](inBlocks: RDD[(Int, InBlock[ID])]): Int = {
    inBlocks.values.map{block => block.srcIds.length}.reduce(_+_)
  }

	private type ArgMap = Map[Symbol,String]

  private def argToMap(keyName: String, value: String): ArgMap =
  {
    val key = Symbol(keyName)
    logStdout(keyName + " flag set to " + value);
    Map(key -> value)
  }

	private def parseArgs(map: ArgMap, list: List[String]) : ArgMap = 
	{
		list match {
			case Nil => map
      case ("--sample" | "-S") :: tail => 
        parseArgs(map ++ argToMap("sampleRatingsFile","true"), tail)
      case ("--implicitAlpha" | "-a" ) :: value :: tail => 
        parseArgs(map ++ argToMap("alpha",value), tail)
      case ("--saveRatings") :: value :: tail => 
        parseArgs(map ++ argToMap("saveRatings",value), tail)
      case ("--hist" | "-H") :: value :: tail =>
        parseArgs(map ++ argToMap("genHistogram",value), tail)
      case ("--CDF" | "-c") :: value :: tail =>
        parseArgs(map ++ argToMap("genCDF",value), tail)
      case ("--als") :: tail =>
        parseArgs(map ++ argToMap("runALS","true"), tail)
			case ("--userBlocks" | "-N") :: value :: tail =>
        parseArgs(map ++ argToMap("userBlocks",value), tail)
			case ("--itemBlocks" | "-M") :: value :: tail =>
        parseArgs(map ++ argToMap("itemBlocks",value), tail)
			case ("--seed" | "-s") :: value :: tail =>
        parseArgs(map ++ argToMap("seed",value), tail)
			case ("--lambda" | "-L") :: value :: tail =>
        parseArgs(map ++ argToMap("regParam",value), tail)
			case ("--rank" | "-f") :: value :: tail =>
        parseArgs(map ++ argToMap("rank",value), tail)
			case ("--ratings" | "-R") :: value :: tail =>
        parseArgs(map ++ argToMap("ratingsFile",value), tail)
			case ("--users" | "-n") :: value :: tail =>
        parseArgs(map ++ argToMap("numUsers",value), tail)
			case ("--items" | "-m") :: value :: tail =>
        parseArgs(map ++ argToMap("numItems",value), tail)
			case ("--obj" | "-O") :: tail =>
        parseArgs(map ++ argToMap("objFile","true"), tail)
			case ("--tol" | "-t") :: value :: tail =>
        parseArgs(map ++ argToMap("tol",value), tail)
			case ("--maxit" | "-k") :: value :: tail =>
        parseArgs(map ++ argToMap("numIters",value), tail)
			case ("--delim" | "-d") :: value :: tail =>
				parseArgs(map ++ argToMap("delim", value), tail)
      case ("--output" | "-o") :: value :: tail =>
				parseArgs(map ++ argToMap("outputPrefix", value), tail)
      case ("--checkpoint" | "-C") :: value :: tail =>
				parseArgs(map ++ argToMap("checkpointDir", value), tail)
			case other :: tail => {
				logStdout("Read non-flag value: " + other)
				parseArgs(map, tail);
			}
		}
	}
  def readRatingsMatrix(
    sc: SparkContext, 
    file: String, 
    delim: String,
    numPartitions: Int): RDD[Rating[Int]] =
  {
    def parseRating(line: String): Rating[Int] =
    {
      val tokens = line.split(delim)
      val u = tokens(0).toInt
      val m = tokens(1).toInt
      val r = tokens(2).toFloat
      Rating[Int](u,m,r)
    }
    sc.textFile(file,numPartitions)
      .map(parseRating)
  }

  def main(args: Array[String]) 
  {
    val conf = new SparkConf()
    val sc = new SparkContext(conf)
		val vars: ArgMap = parseArgs(Map(),args.toList);
    val userBlocks: Int = if (vars.contains('userBlocks)) vars('userBlocks).toInt else 10
    val itemBlocks: Int = if (vars.contains('itemBlocks)) vars('itemBlocks).toInt else 10
    val rank: Int = if (vars.contains('rank)) vars('rank).toInt else 10
    val numIters: Int = if (vars.contains('numIters)) vars('numIters).toInt else 20
    val regParam: Double = if (vars.contains('regParam)) vars('regParam).toDouble else 0.01
    val delim: String = if (vars.contains('delim)) vars('delim) else ","
    val seed: Long = if (vars.contains('seed)) vars('seed).toLong else 0l
    val implicitPrefs: Boolean = vars.contains('implicitAlpha)
    val alpha: Double = if (vars.contains('implicitAlpha)) vars('implicitAlpha).toDouble else 0.0
    if (vars.contains('checkpointDir))
      sc.setCheckpointDir(vars('checkpointDir))

    val ratings: RDD[Rating[Int]] = {
      if (vars.contains('numUsers))
      {
        val numUsers: Int = vars('numUsers).toInt
        logStdout("Reading up to " + numUsers + " from "  + vars('ratingsFile))
        readRatingsMatrix(sc, vars('ratingsFile), delim, userBlocks)
        .filter{case Rating(u,m,r) => (u <= numUsers) }
      } else {
        logStdout("Reading " + vars('ratingsFile))
        readRatingsMatrix(sc, vars('ratingsFile), delim, userBlocks)
      }
    }.cache()
    logStdout(s"Read ratings matrix with ${ratings.count} ratings")
    val (userFactors,itemFactors) = {
    /*if (implicitPrefs)*/
    /*  PNCG.trainImplicit(ratings,rank,numIters,regParam,userBlocks,alpha,seed,runPNCG)*/
    /*else*/
      PNCG.train(
        ratings,
        rank,
        userBlocks,
        userBlocks,
        numIters,
        regParam,
        implicitPrefs,
        alpha,
        nonnegative = false,
        seed = seed)
        /*intermediateRDDStorageLevel = StorageLevel.MEMORY_AND_DISK,*/
        /*finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,*/
        /*checkpointInterval: Int = 10,*/
        /*seed: Long = 0L)(*/
      /*ratings,rank,numIters,regParam,userBlocks,seed)*/
    }

  }
}
