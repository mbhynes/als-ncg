/*
 * Solver.scala
 *
 * Contains the LeastSquares/Cholesky solver routines
 * used in ALS.
 *
 * =================================================
 * Author: Xianggrui Meng
 * License: GPL 3
 * Creation Date: Tue 20 Oct 2015 09:25:49 PM EDT
 * Last Modified: Tue 20 Oct 2015 09:30:15 PM EDT
 * =================================================
 */

package himrod.ncg

import java.util.Random
import java.{util => ju}
import java.io.IOException

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

import himrod.ncg.utils._


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
