package task3;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.*;

public class KMeansCluster {

    private static final int K = 3;
    private static final int MAX_ITERATIONS = 20;
    private static final double CONVERGENCE_THRESHOLD = 0.0001;
    private static final long RANDOM_SEED = 42L;
    private static final String DEFAULT_DATA_BASE = "/user/root/final_exp/exp1";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
            .setAppName("Task3_KMeansClustering");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }

        String task1OutputBase = "/user/" + hdfsUser + "/final_exp/exp1/output";
        String task3OutputBase = "/user/" + hdfsUser + "/final_exp/exp3/output";
        String inputPath = args.length > 0 ? args[0] : task1OutputBase + "/user_features.csv";
        String outputBase = args.length > 1 ? args[1] : task3OutputBase;
        String trainPath = args.length > 2 ? args[2] : DEFAULT_DATA_BASE + "/train.csv";

        String outputCentersPath = outputBase + "/cluster_centers.txt";
        String outputLabelsPath = outputBase + "/user_cluster_labels.csv";
        System.out.println("input:  " + inputPath);
        System.out.println("train:  " + trainPath);
        System.out.println("output: " + outputBase);

        // 仅使用训练集用户
        JavaPairRDD<Long, Boolean> trainUserIds = sc.textFile(trainPath)
            .filter(line -> !line.startsWith("user_id") && !line.trim().isEmpty())
            .map(line -> line.split(",")[0].trim())
            .filter(uid -> !uid.isEmpty())
            .mapToPair(uid -> new Tuple2<>(Long.parseLong(uid), true))
            .reduceByKey((a, b) -> a);

        long trainUserCount = trainUserIds.count();
        System.out.println("训练集用户数: " + trainUserCount);

        JavaPairRDD<Long, double[]> userVectors = sc.textFile(inputPath)
            .filter(line -> !line.trim().isEmpty())
            .mapToPair(line -> {
                String[] parts = line.split(",");
                if (parts.length < 4) {
                    return null;
                }
                try {
                    long userId = Long.parseLong(parts[0].trim());
                    double[] vector = new double[3];
                    vector[0] = Double.parseDouble(parts[1].trim());
                    vector[1] = Double.parseDouble(parts[2].trim());
                    vector[2] = Double.parseDouble(parts[3].trim());
                    return new Tuple2<>(userId, vector);
                } catch (NumberFormatException e) {
                    return null;
                }
            })
            .filter(tuple -> tuple != null)
            .join(trainUserIds)
            .mapToPair(tuple -> new Tuple2<>(tuple._1, tuple._2._1));

        long userCount = userVectors.count();
        System.out.println("参与聚类的训练集用户数: " + userCount);
        if (userCount < K) {
            throw new IllegalStateException("用户数少于聚类数 K=" + K);
        }

        List<double[]> centers = initializeCentersRandomly(userVectors.values(), K);
        System.out.println("初始聚类中心:");
        for (int i = 0; i < centers.size(); i++) {
            double[] c = centers.get(i);
            System.out.printf("中心%d: [%.4f, %.4f, %.4f]\n", i, c[0], c[1], c[2]);
        }

        Broadcast<List<double[]>> broadcastCenters = sc.broadcast(centers);

        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            System.out.println("\n========== 迭代 " + (iter + 1) + " ==========");

            final Broadcast<List<double[]>> iterCenters = broadcastCenters;
            JavaPairRDD<Long, Integer> assignments = userVectors
                .mapToPair(tuple -> new Tuple2<>(
                    tuple._1,
                    nearestCluster(tuple._2, iterCenters.value())
                ));

            JavaPairRDD<Integer, Tuple2<double[], Integer>> clusterSums = assignments
                .join(userVectors)
                .mapToPair(tuple -> {
                    int clusterId = tuple._2._1;
                    double[] vector = tuple._2._2;
                    return new Tuple2<>(clusterId, new Tuple2<>(vector, 1));
                })
                .reduceByKey((a, b) -> {
                    double[] sum = new double[3];
                    sum[0] = a._1[0] + b._1[0];
                    sum[1] = a._1[1] + b._1[1];
                    sum[2] = a._1[2] + b._1[2];
                    return new Tuple2<>(sum, a._2 + b._2);
                });

            Map<Integer, double[]> centerMap = new HashMap<>();
            for (Tuple2<Integer, Tuple2<double[], Integer>> tuple : clusterSums.collect()) {
                int clusterId = tuple._1;
                double[] sum = tuple._2._1;
                int count = tuple._2._2;
                double[] center = new double[3];
                for (int i = 0; i < 3; i++) {
                    center[i] = sum[i] / count;
                }
                centerMap.put(clusterId, center);
            }

            List<double[]> newCenters = mergeCenters(centers, centerMap, K);

            double maxShift = 0.0;
            for (int i = 0; i < K; i++) {
                maxShift = Math.max(maxShift, euclideanDistance(centers.get(i), newCenters.get(i)));
            }
            System.out.printf("最大中心点偏移: %.6f\n", maxShift);

            broadcastCenters.destroy();
            centers = newCenters;
            broadcastCenters = sc.broadcast(centers);

            if (maxShift < CONVERGENCE_THRESHOLD) {
                System.out.println("已收敛，停止迭代");
                break;
            }
        }

        final Broadcast<List<double[]>> finalCentersBC = broadcastCenters;
        JavaPairRDD<Long, Integer> finalAssignments = userVectors
            .mapToPair(tuple -> new Tuple2<>(
                tuple._1,
                nearestCluster(tuple._2, finalCentersBC.value())
            ));

        JavaRDD<String> centersOutput = sc.parallelize(centers).zipWithIndex()
            .map(tuple -> {
                double[] center = tuple._1;
                long clusterId = tuple._2;
                return String.format("%d\t%.4f,%.4f,%.4f",
                    clusterId, center[0], center[1], center[2]);
            });
        centersOutput.coalesce(1).saveAsTextFile(outputCentersPath);
        System.out.println("聚类中心已保存: " + outputCentersPath);

        JavaRDD<String> labelsOutput = finalAssignments
            .map(tuple -> tuple._1 + "," + tuple._2);
        labelsOutput.saveAsTextFile(outputLabelsPath);
        System.out.println("用户聚类标签已保存: " + outputLabelsPath);

        System.out.println("\n========== 任务三完成 ==========");
        System.out.println("最终聚类中心:");
        for (int i = 0; i < centers.size(); i++) {
            double[] c = centers.get(i);
            System.out.printf("簇%d: [%.4f, %.4f, %.4f]\n", i, c[0], c[1], c[2]);
        }

        Map<Integer, Long> clusterCounts = finalAssignments
            .mapToPair(tuple -> new Tuple2<>(tuple._2, 1L))
            .reduceByKey(Long::sum)
            .collectAsMap();

        System.out.println("\n各簇用户数量:");
        for (int i = 0; i < K; i++) {
            System.out.printf("簇%d: %d 个用户\n", i, clusterCounts.getOrDefault(i, 0L));
        }

        System.out.println("\n用户聚类标签示例 (前20条):");
        for (String sample : labelsOutput.take(20)) {
            System.out.println(sample);
        }

        broadcastCenters.destroy();
        sc.close();
    }

    private static List<double[]> mergeCenters(
            List<double[]> oldCenters,
            Map<Integer, double[]> updated,
            int k) {
        List<double[]> merged = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            if (updated.containsKey(i)) {
                merged.add(updated.get(i));
            } else {
                merged.add(copyVector(oldCenters.get(i)));
            }
        }
        return merged;
    }

    private static double[] copyVector(double[] src) {
        double[] copy = new double[src.length];
        System.arraycopy(src, 0, copy, 0, src.length);
        return copy;
    }

    private static int nearestCluster(double[] vector, List<double[]> centerList) {
        int nearestCluster = 0;
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < centerList.size(); i++) {
            double dist = euclideanDistance(vector, centerList.get(i));
            if (dist < minDistance) {
                minDistance = dist;
                nearestCluster = i;
            }
        }
        return nearestCluster;
    }

    private static List<double[]> initializeCentersRandomly(JavaRDD<double[]> vectors, int k) {
        List<double[]> samples = vectors.takeSample(false, k, RANDOM_SEED);
        List<double[]> centers = new ArrayList<>(samples);
        while (centers.size() < k) {
            centers.add(copyVector(centers.get(centers.size() - 1)));
        }
        return centers;
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
