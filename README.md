# ALS-NCG
This repo contains the source code for the Apache Spark implementation of the [ALS-NCG](http://arxiv.org/abs/1508.03110) algorithm for use as a Spark package. The algorithm uses the Alternating Least Squares (ALS) algorithm as a nonlinear preconditioner to the Nonlinear Conjugate Gradient (NCG) algorithm for solving low rank matrix factorization problem. This document gives an introduction to the code structure and how to call the routines on Spark. Currently, only the explicit objective function is supported, and not the implicit version.

## Code Structure & Use in Spark Programs
The ALS-NCG algorithm is implemented using the existing [ALS] (https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/recommendation/ALS.scala) algorithm implementation in Spark (the original code is described in the [ALS-NCG](http://arxiv.org/abs/1508.03110) paper). The vector operations in the NCG algorithm are built alongside the ALS operations. % on Resiliend Distributed Datasets (RDDs).

The training routine is contained in the `NCG` object and called `trainPNCG` (the PNCG acronym stands for *preconditioned* NCG). This routine returns RDDs of type `RDD[(ID, Array[Float])]`, where `ID` is either an integer or long, and the floating point array is the feature vector of a given rank. 
There are other routines in the object, `trainALS` and `trainNCG`, that will run the standalone ALS and NCG algorithms, respectively. These routines exist merely for comparison. 

The `trainPNCG` routine has the following specification:
```scala
  def trainPNCG[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1e-2,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      nonnegative: Boolean = false,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval: Int = 10,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])])
```

To use ALS-NCG in a Spark program, once the package has been imported, the routine can be called from within the `NCG` object if the ratings file is specified.

### Running ALS-NCG from the Commandline
The `NCG` object contains a `main` routine that parses commandline arguments for parameter flags if the ALS-NCG algorithm is run using `spark-submit`, which is a typical use case for performance testing. The relevant flags for this are in the following code snippet:
```scala
private def parseArgs(map: ArgMap, list: List[String]) : ArgMap = 
{
	list match {
		case Nil => map

	// flag to run standalone ALS
	case ("--als") :: tail =>
		parseArgs(map ++ argToMap("runALS","true"), tail)

	// flag to run standalone NCG
	case ("--ncg") :: tail =>
		parseArgs(map ++ argToMap("runNCG","true"), tail)

	// specify the number of blocks for user feature vectors
	case ("--userBlocks" | "-N") :: value :: tail =>
		parseArgs(map ++ argToMap("userBlocks",value), tail)

	// specify the number of blocks for item feature vectors
	case ("--itemBlocks" | "-M") :: value :: tail =>
		parseArgs(map ++ argToMap("itemBlocks",value), tail)

	// set the seed for initialization of feature vectors
	case ("--seed" | "-s") :: value :: tail =>
		parseArgs(map ++ argToMap("seed",value), tail)

	// set the value of the regularization parameter 
	case ("--lambda" | "-L") :: value :: tail =>
		parseArgs(map ++ argToMap("regParam",value), tail)

	// set rank of the feature space
	case ("--rank" | "-f") :: value :: tail =>
		parseArgs(map ++ argToMap("rank",value), tail)

	// specify the path to the ratings file 
	case ("--ratings" | "-R") :: value :: tail =>
		parseArgs(map ++ argToMap("ratingsFile",value), tail)

	// specify the delimiter in the ratings file (default is ",")
	case ("--delim" | "-d") :: value :: tail =>
		parseArgs(map ++ argToMap("delim", value), tail)

	// set the number of iterations to run
	case ("--maxit" | "-k") :: value :: tail =>
		parseArgs(map ++ argToMap("numIters",value), tail)

	// set the SparkContext checkpoint directory
	case ("--checkpoint" | "-C") :: value :: tail =>
		parseArgs(map ++ argToMap("checkpointDir", value), tail)
```

If neither the `--als` or `--ncg` flags are given, the ALS-NCG algorithm will be run.
The ratings file is expected to have the form of `i,j,R_{ij}` on each line, where the delimeter can be specified as any string.

When invoking the ALS-NCG algorithm from the commandline with `spark-submit`, the arguments are passed as strings appended to the commandline after the `NCG` jar pathname. An example invocation is the following bash snippet, where the relevant environment variables have been set (i.e., `$JAR` is the path to the packaged `NCG` jar on your system):

```sh
spark-submit \                                                                                
	--deploy-mode $SPARK_DEPLOY_MODE \
	--name $NAME \
	--class $CLASS \
	--master $SPARK_MASTER_URL \
	--driver-memory $SPARK_DRIVER_MEM \
	--executor-memory $SLAVE_MEM \
	--executor-cores "$executor_cores" \
	--conf spark.executor.extraJavaOptions=-Djava.io.tmpdir=$JAVA_IO_TMPDIR \
	$JAR \
	--seed 0 \
	--lambda 0.01 \
	--rank 10 \
	--userBlocks 16 \
	--ratings ratings_file
```
