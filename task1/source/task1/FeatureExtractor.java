package task1;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.List;

public class FeatureExtractor {

    private static final double SECONDS_PER_DAY = 86400.0;
    private static final String DEFAULT_DATA_BASE = "/user/root/final_exp/exp1";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
            .setAppName("Task1_FeatureExtraction");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }

        String trainPath = args.length > 0 ? args[0] : DEFAULT_DATA_BASE + "/train.csv";
        String testPath = args.length > 1 ? args[1] : DEFAULT_DATA_BASE + "/test.csv";
        String outputBase = args.length > 2 ? args[2]
            : "/user/" + hdfsUser + "/final_exp/exp1/output";
        String outputLogPath = outputBase + "/clean_behavior_log.csv";
        String outputFeaturePath = outputBase + "/user_features.csv";

        System.out.println("train:  " + trainPath);
        System.out.println("test:   " + testPath);
        System.out.println("output: " + outputBase);

        // 训练集、测试集分别清洗
        JavaRDD<BehaviorLog> trainLogs = parseAndClean(sc.textFile(trainPath));
        JavaRDD<BehaviorLog> testLogs = parseAndClean(sc.textFile(testPath));
        JavaRDD<BehaviorLog> allLogs = trainLogs.union(testLogs);

        long trainCount = trainLogs.count();
        long testCount = testLogs.count();
        long validCount = allLogs.count();
        System.out.println("训练集有效日志: " + trainCount);
        System.out.println("测试集有效日志: " + testCount);
        System.out.println("全量有效日志:   " + validCount);

        // 截止时间与极值基于 train + test 全量
        long globalMaxTimestamp = allLogs.map(log -> log.timestamp).reduce(Math::max);
        System.out.println("全局最大时间戳: " + globalMaxTimestamp);

        JavaPairRDD<Long, UserAgg> trainAggRdd = aggregateByUser(trainLogs);
        JavaPairRDD<Long, UserAgg> testAggRdd = aggregateByUser(testLogs);
        // 分别聚合后合并；同一用户若同时出现在 train/test，需 merge
        JavaPairRDD<Long, UserAgg> allAggRdd = trainAggRdd.union(testAggRdd)
            .reduceByKey(UserAgg::merge);

        JavaRDD<UserFeatures> userFeaturesRdd = allAggRdd.map(tuple -> {
            long userId = tuple._1;
            UserAgg agg = tuple._2;
            double recencyDays = (globalMaxTimestamp - agg.maxTimestamp) / SECONDS_PER_DAY;
            return new UserFeatures(userId, recencyDays, agg.activityCount, agg.valueScore);
        });

        long userCount = userFeaturesRdd.count();
        System.out.println("用户数量(全量): " + userCount);

        double f1Min = userFeaturesRdd.map(f -> f.recencyDays).reduce(Math::min);
        double f1Max = userFeaturesRdd.map(f -> f.recencyDays).reduce(Math::max);
        double f2Min = userFeaturesRdd.map(f -> (double) f.activityCount).reduce(Math::min);
        double f2Max = userFeaturesRdd.map(f -> (double) f.activityCount).reduce(Math::max);
        double f3Min = userFeaturesRdd.map(f -> f.valueScore).reduce(Math::min);
        double f3Max = userFeaturesRdd.map(f -> f.valueScore).reduce(Math::max);

        System.out.printf("F1(时效/天) 范围: [%.4f, %.4f]%n", f1Min, f1Max);
        System.out.printf("F2(活跃度)  范围: [%.0f, %.0f]%n", f2Min, f2Max);
        System.out.printf("F3(价值分)  范围: [%.4f, %.4f]%n", f3Min, f3Max);

        final double[] bounds = {f1Min, f1Max, f2Min, f2Max, f3Min, f3Max};
        JavaRDD<String> normalizedRdd = userFeaturesRdd.map(feature -> {
            double normF1 = minMaxNormalize(feature.recencyDays, bounds[0], bounds[1]);
            double normF2 = minMaxNormalize(feature.activityCount, bounds[2], bounds[3]);
            double normF3 = minMaxNormalize(feature.valueScore, bounds[4], bounds[5]);
            return String.format("%d,%.4f,%.4f,%.4f",
                feature.userId, normF1, normF2, normF3);
        });

        allLogs.map(BehaviorLog::toString).saveAsTextFile(outputLogPath);
        System.out.println("已保存: " + outputLogPath);

        normalizedRdd.saveAsTextFile(outputFeaturePath);
        System.out.println("已保存: " + outputFeaturePath);

        System.out.println("\n========== 任务一完成 ==========");
        System.out.println("clean_behavior_log.csv 行数: " + validCount);
        System.out.println("user_features.csv 行数: " + userCount);

        System.out.println("\n归一化特征预览 (前10行):");
        List<String> samples = normalizedRdd.take(10);
        for (String sample : samples) {
            System.out.println(sample);
        }

        sc.close();
    }

    private static JavaRDD<BehaviorLog> parseAndClean(JavaRDD<String> rawRdd) {
        JavaRDD<String> dataRdd = rawRdd.filter(line ->
            !line.startsWith("user_id") && !line.trim().isEmpty()
        );

        return dataRdd
            .map(line -> {
                String[] parts = line.split(",");
                if (parts.length < 5) {
                    return null;
                }
                try {
                    return new BehaviorLog(
                        Long.parseLong(parts[0].trim()),
                        Long.parseLong(parts[1].trim()),
                        Integer.parseInt(parts[2].trim()),
                        parts[3].trim().toLowerCase(),
                        Long.parseLong(parts[4].trim())
                    );
                } catch (NumberFormatException e) {
                    return null;
                }
            })
            .filter(log -> log != null)
            .filter(log -> log.behaviorType.equals("pv")
                || log.behaviorType.equals("cart")
                || log.behaviorType.equals("fav")
                || log.behaviorType.equals("buy"));
    }

    private static JavaPairRDD<Long, UserAgg> aggregateByUser(JavaRDD<BehaviorLog> logsRdd) {
        return logsRdd
            .mapToPair(log -> new Tuple2<>(log.userId, log))
            .aggregateByKey(
                new UserAgg(),
                UserAgg::add,
                UserAgg::merge);
    }

    private static double minMaxNormalize(double value, double min, double max) {
        if (max == min) {
            return 0.5;
        }
        return (value - min) / (max - min);
    }

    private static int behaviorScore(String behaviorType) {
        switch (behaviorType) {
            case "buy": return 4;
            case "cart": return 3;
            case "fav": return 2;
            case "pv": return 1;
            default: return 0;
        }
    }

    static class UserAgg implements java.io.Serializable {
        private static final long serialVersionUID = 1L;

        long maxTimestamp = Long.MIN_VALUE;
        int activityCount = 0;
        double valueScore = 0.0;

        UserAgg add(BehaviorLog log) {
            maxTimestamp = Math.max(maxTimestamp, log.timestamp);
            activityCount++;
            valueScore += behaviorScore(log.behaviorType);
            return this;
        }

        UserAgg merge(UserAgg other) {
            maxTimestamp = Math.max(maxTimestamp, other.maxTimestamp);
            activityCount += other.activityCount;
            valueScore += other.valueScore;
            return this;
        }
    }
}
