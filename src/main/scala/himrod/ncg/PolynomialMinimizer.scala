/*
 * 
 * src/main/scala/himrod/ncg/PolynomialMinimizer.scala
 * =================================================
 * Author: Michael B Hynes, mbhynes@uwaterloo.ca
 * License: GPL 3
 * Creation Date: Mon 15 Feb 2016 05:53:23 PM EST
 * Last Modified: Mon 15 Feb 2016 05:55:52 PM EST
 * =================================================
 */

package himrod.ncg

private class PolynomialMinimizer(funcCoeffs: Array[Float]) 
{
  val VERBOSE = false
  val degree = funcCoeffs.length - 1
  val gradCoeffs = funcCoeffs.zipWithIndex.tail.map{case (c,n) => n*c}
  val hessCoeffs = gradCoeffs.zipWithIndex.tail.map{case (c,n) => n*c}

  private def logStdout(msg: String): Unit = {
		val time: Long = System.currentTimeMillis;
		println(time + ": " + msg);
	}
  private def func(x: Float): Float = funcCoeffs.foldRight(0f)((a_n,sum) => a_n + x * sum)
  private def grad(x: Float): Float = gradCoeffs.foldRight(0f)((a_n,sum) => a_n + x * sum)
  private def hess(x: Float): Float = hessCoeffs.foldRight(0f)((a_n,sum) => a_n + x * sum)

  // compute minimum around x0 using Newton's method
  // We find zeros of the gradient, since we are guaranteed that
  // the polynomial f(x) is decreasing at x = 0 for our problem
  def minimize(x0: Float, tol: Float, maxIters: Int): (Float,Float) = 
  {
    var x = x0;
    var g = grad(x)
    var k = 1;
    if (VERBOSE) logStdout(s"Rootfinder: $k: $x: $g: ${func(x)}")
    while ((math.abs(g) > tol) && (k <= maxIters)) {
      g = grad(x)
      x -= g / hess(x)
      k += 1
      if (VERBOSE) logStdout(s"Rootfinder: $k: $x: $g: ${func(x)}")
    }
    (x,func(x))
  }

  def minimize(xs: Array[Float], tol: Float, maxIters: Int): (Float,Float) = 
  {
    val str = new StringBuilder(10 * xs.length)
    funcCoeffs.foreach{x => str.append(x.toString + ",")}
    if (VERBOSE) logStdout(s"PolynomialMinimizer: coeff: ${str.mkString}")
    /*str.clear*/
    /*stepSizes.foreach{x => str.append(x.toString + ",")}*/
    /*logStdout(s"minimize: stepsizes for minima: ${str.mkString}")*/
    val res = xs.map{x => minimize(x,tol,maxIters)}
    val stepSizes = res.map{x => x._1}
    val minima = res.map{x => x._2}
    /*val str = new StringBuilder(10 * xs.length)*/
    /*minima.foreach{x => str.append(x.toString + ",")}*/
    /*logStdout(s"minimize: found local minima: ${str.mkString}")*/
    /*str.clear*/
    /*stepSizes.foreach{x => str.append(x.toString + ",")}*/
    /*logStdout(s"minimize: stepsizes for minima: ${str.mkString}")*/
    val step = stepSizes(minima.zipWithIndex.min._2)
    (step,func(step))
  }
}
