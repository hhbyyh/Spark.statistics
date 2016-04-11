/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.mllib.stat.test

import org.apache.commons.math3.util.FastMath
import org.apache.spark.rdd.RDD
import org.apache.commons.math3.stat.inference.{KolmogorovSmirnovTest => CommonMathKolmogorovSmirnovTest}

/**
 * We just found out KolmogorovSmirnovTest was included in Spark 1.6.
 * Instead of providing a new implementation, it better aligns with
 * users' interests if we can provide improvement or new functions based
 * on the Spark version. If you find something potentially useful yet missing
 * from Spark, please go ahead and create an issue in the project.
 *
 */
object KolmogorovSmirnovTest {

  // Null hypothesis for the type of KS test to be included in the result.
  object NullHypothesis extends Enumeration {
    type NullHypothesis = Value
    val OneSampleTwoSided = Value("Sample follows theoretical distribution")
    val TwoSampleTwoSided = Value("Both samples follow same distribution")
  }

  def ksSum (t: Double, tolerance: Double, maxIterations: Int): Double = {
    if (t == 0.0) {
      return 0.0
    }
    val x: Double = -2 * t * t
    var sign: Int = -1
    var i: Long = 1
    var partialSum: Double = 0.5d
    var delta: Double = 1
    while (delta > tolerance && i < maxIterations) {
      delta = FastMath.exp(x * i * i)
      partialSum += sign * delta
      sign *= -1
      i += 1
    }
    partialSum * 2
  }


  /**
   * Implements a two-sample, two-sided Kolmogorov-Smirnov test, which tests if the 2 samples
   * come from the same distribution
   * @param data1 `RDD[Double]` first sample of data
   * @param data2 `RDD[Double]` second sample of data
   * @return [[org.apache.spark.mllib.stat.test.KolmogorovSmirnovTestResult]] with the test
   *        statistic, p-value, and appropriate null hypothesis
   */
  def testTwoSamples(data1: RDD[Double], data2: RDD[Double]): KolmogorovSmirnovTestResult = {
    val n1 = data1.count().toDouble
    val n2 = data2.count().toDouble
    // identifier for sample 1, needed after co-sort
    val isSample1 = true
    // combine identified samples
    val unionedData = data1.map((_, isSample1)).union(data2.map((_, !isSample1)))
    // co-sort and operate on each partition, returning local extrema to the driver
    val localData = unionedData.sortByKey().mapPartitions(
      searchTwoSampleCandidates(_, n1, n2)
    ).collect()
    // result: global extreme
    val ksStat = searchTwoSampleStatistic(localData, n1 * n2)
    evalTwoSampleP(ksStat, n1.toInt, n2.toInt)
  }

  /**
   * Calculates maximum distance candidates and counts of elements from each sample within one
   * partition for the two-sample, two-sided Kolmogorov-Smirnov test implementation. Function
   * is package private for testing convenience.
   * @param partData `Iterator[(Double, Boolean)]` the data in 1 partition of the co-sorted RDDs,
   *                each element is additionally tagged with a boolean flag for sample 1 membership
   * @param n1 `Double` sample 1 size
   * @param n2 `Double` sample 2 size
   * @return `Iterator[(Double, Double, Double)]` where the first element is an unadjusted minimum
   *        distance, the second is an unadjusted maximum distance (both of which will later
   *        be adjusted by a constant to account for elements in prior partitions), and the third is
   *        a count corresponding to the numerator of the adjustment constant coming from this
   *        partition. This last value, the numerator of the adjustment constant, is calculated as
   *        |sample 2| * |sample 1 in partition| - |sample 1| * |sample 2 in partition|. This comes
   *        from the fact that when we adjust for prior partitions, what we are doing is
   *        adding the difference of the fractions (|prior elements in sample 1| / |sample 1| -
   *        |prior elements in sample 2| / |sample 2|). We simply keep track of the numerator
   *        portion that is attributable to each partition so that following partitions can
   *        use it to cumulatively adjust their values.
   */
  private[stat] def searchTwoSampleCandidates(
                                               partData: Iterator[(Double, Boolean)],
                                               n1: Double,
                                               n2: Double): Iterator[(Double, Double, Double)] = {
    // fold accumulator: local minimum, local maximum, index for sample 1, index for sample2
    case class ExtremaAndRunningIndices(min: Double, max: Double, ix1: Int, ix2: Int)
    val initAcc = ExtremaAndRunningIndices(Double.MaxValue, Double.MinValue, 0, 0)
    // traverse the data in the partition and calculate distances and counts
    val pResults = partData.foldLeft(initAcc) { case (acc, (v, isSample1)) =>
      val (add1, add2) = if (isSample1) (1, 0) else (0, 1)
      val cdf1 = (acc.ix1 + add1) / n1
      val cdf2 = (acc.ix2 + add2) / n2
      val dist = cdf1 - cdf2
      ExtremaAndRunningIndices(
        math.min(acc.min, dist),
        math.max(acc.max, dist),
        acc.ix1 + add1, acc.ix2 + add2
      )
    }
    // If partition has no data, then pResults will match the fold accumulator
    // we must filter this out to avoid having the statistic spoiled by the accumulation values
    val results = if (pResults == initAcc) {
      Array[(Double, Double, Double)]()
    } else {
      Array((pResults.min, pResults.max, (pResults.ix1 + 1) * n2 - (pResults.ix2 + 1) * n1))
    }
    results.iterator
  }

  /**
   * Adjust candidate extremes by the appropriate constant. The resulting maximum corresponds to
   * the two-sample, two-sided Kolmogorov-Smirnov test. Function is package private for testing
   * convenience.
   * @param localData `Array[(Double, Double, Double)]` contains the candidate extremes from each
   *                 partition, along with the numerator for the necessary constant adjustments
   * @param n `Double` The denominator in the constant adjustment (i.e. (size of sample 1 ) * (size
   *         of sample 2))
   * @return The two-sample, two-sided Kolmogorov-Smirnov statistic
   */
  private[stat] def searchTwoSampleStatistic(localData: Array[(Double, Double, Double)], n: Double)
  : Double = {
    // maximum distance and numerator for constant adjustment
    val initAcc = (Double.MinValue, 0.0)
    // adjust differences based on the number of elements preceding it, which should provide
    // the correct distance between the 2 empirical CDFs
    val results = localData.foldLeft(initAcc) { case ((prevMax, prevCt), (minCand, maxCand, ct)) =>
      val adjConst = prevCt / n
      val dist1 = math.abs(minCand + adjConst)
      val dist2 = math.abs(maxCand + adjConst)
      val maxVal = Array(prevMax, dist1, dist2).max
      (maxVal, prevCt + ct)
    }
    results._1
  }

  private def evalTwoSampleP(ksStat: Double, n: Int, m: Int): KolmogorovSmirnovTestResult = {
    val pval = new CommonMathKolmogorovSmirnovTest().approximateP(ksStat, n, m)
    new KolmogorovSmirnovTestResult(pval, ksStat, NullHypothesis.TwoSampleTwoSided.toString)
  }
}

