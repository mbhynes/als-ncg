name := "ncg"

version := "1.0"

compileOrder := CompileOrder.Mixed

parallelExecution in Test := false

scalaVersion := "2.10.4"

organization := "himrod"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "1.5.1" % "provided",
	"org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided",
	"org.jblas" % "jblas" % "1.2.4",
	"org.scalatest" % "scalatest_2.10" % "2.0" % "test",
	"com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

resolvers ++= Seq(
	"Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
	"Sonatype OSS Releases" at "https://oss.sonatype.org/content/repositories/releases"
)
