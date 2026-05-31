package task1;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.List;

public class FeatureExtractor {

    private static final double SECONDS_PER_DAY = 86400.0;

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
            .setAppName("Task1_FeatureExtraction");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }
        String inputPath = args.length > 0 ? args[0]
            : "/user/root/final_exp/exp1/train.csv";
        String outputBase = args.length > 1 ? args[1]
            : "/user/" + hdfsUser + "/final_exp/exp1/output";
        String outputLogPath = outputBase + "/clean_behavior_log.csv";
        String outputFeaturePath = outputBase + "/user_features.csv";
        System.out.println("input:  " + inputPath);
        System.out.println("output: " + outputBase);

        JavaRDD<String> rawRdd = sc.textFile(inputPath);

        JavaRDD<String> dataRdd = rawRdd.filter(line ->
            !line.startsWith("user_id") && !line.trim().isEmpty()
        );

        System.out.println("原始日志行数: " + dataRdd.count());

        JavaRDD<BehaviorLog> logsRdd = dataRdd
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

        long validCount = logsRdd.count();
        System.out.println("有效日志行数: " + validCount);

        long globalMaxTimestamp = logsRdd.map(log -> log.timestamp).reduce(Math::max);
        System.out.println("全局最大时间戳: " + globalMaxTimestamp);

        // aggregateByKey: 按 user_id 聚合 F1/F2/F3 原始值
        JavaPairRDD<Long, UserAgg> userAggRdd = logsRdd
            .mapToPair(log -> new Tuple2<>(log.userId, log))
            .aggregateByKey(
                new UserAgg(),
                UserAgg::add,
                UserAgg::merge);

        JavaRDD<UserFeatures> userFeaturesRdd = userAggRdd.map(tuple -> {
            long userId = tuple._1;
            UserAgg agg = tuple._2;
            double recencyDays = (globalMaxTimestamp - agg.maxTimestamp) / SECONDS_PER_DAY;
            return new UserFeatures(userId, recencyDays, agg.activityCount, agg.valueScore);
        });

        long userCount = userFeaturesRdd.count();
        System.out.println("用户数量: " + userCount);

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

        logsRdd.map(BehaviorLog::toString).saveAsTextFile(outputLogPath);
        System.out.println("已保存: " + outputLogPath);

        normalizedRdd.saveAsTextFile(outputFeaturePath);
        System.out.println("已保存: " + outputFeaturePath);

        System.out.println("\n========== 任务一完成 ==========");
        System.out.println("clean_behavior_log.csv 行数: " + logsRdd.count());
        System.out.println("user_features.csv 行数: " + normalizedRdd.count());

        System.out.println("\n归一化特征预览 (前10行):");
        List<String> samples = normalizedRdd.take(10);
        for (String sample : samples) {
            System.out.println(sample);
        }

        sc.close();
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
