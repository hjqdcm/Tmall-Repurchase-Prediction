package task1;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.*;

public class FeatureExtractor {
    public static void main(String[] args) {
        // 初始化SparkContext
        SparkConf conf = new SparkConf()
            .setAppName("Task1_FeatureExtraction");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }
        // 平台默认 HDFS 路径；本地测试可传参：args[0]=输入, args[1]=输出目录
        String inputPath = args.length > 0 ? args[0]
            : "/user/root/final_exp/exp1/train.csv";
        String outputBase = args.length > 1 ? args[1]
            : "/user/" + hdfsUser + "/final_exp/exp1/output";
        String outputLogPath = outputBase + "/clean_behavior_log.csv";
        String outputFeaturePath = outputBase + "/user_features.csv";
        System.out.println("input:  " + inputPath);
        System.out.println("output: " + outputBase);
        
        // ==================== 步骤1: 读取数据 ====================
        JavaRDD<String> rawRdd = sc.textFile(inputPath);
        
        JavaRDD<String> dataRdd = rawRdd.filter(line -> 
            !line.startsWith("user_id") && !line.trim().isEmpty()
        );
        
        long totalLines = dataRdd.count();
        System.out.println("原始日志行数: " + totalLines);
        
        // ==================== 步骤2: 解析并过滤 ====================
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
            .filter(log -> {
                return log.behaviorType.equals("pv") || 
                       log.behaviorType.equals("cart") || 
                       log.behaviorType.equals("fav") || 
                       log.behaviorType.equals("buy");
            });
        
        long validCount = logsRdd.count();
        System.out.println("有效日志行数: " + validCount);
        
        // ==================== 步骤3: 按用户聚合 ====================
        JavaPairRDD<Long, Iterable<BehaviorLog>> userLogsGrouped = logsRdd
            .mapToPair(log -> new Tuple2<>(log.userId, log))
            .groupByKey();
        
        JavaRDD<UserFeatures> userFeaturesRdd = userLogsGrouped
            .map(tuple -> {
                long userId = tuple._1;
                Iterable<BehaviorLog> logs = tuple._2;
                
                int totalCount = 0;
                int buyCount = 0;
                int pvCount = 0;
                
                for (BehaviorLog log : logs) {
                    totalCount++;
                    if (log.behaviorType.equals("buy")) {
                        buyCount++;
                    } else if (log.behaviorType.equals("pv")) {
                        pvCount++;
                    }
                }
                
                double conversionRate = pvCount == 0 ? 0.0 : (double) buyCount / pvCount;
                
                return new UserFeatures(userId, totalCount, buyCount, conversionRate);
            });
        
        long userCount = userFeaturesRdd.count();
        System.out.println("用户数量: " + userCount);
        
        // ==================== 步骤4: 计算最大最小值 ====================
        List<UserFeatures> featureList = userFeaturesRdd.collect();
        
        int minF1 = Integer.MAX_VALUE, maxF1 = Integer.MIN_VALUE;
        int minF2 = Integer.MAX_VALUE, maxF2 = Integer.MIN_VALUE;
        double minF3 = Double.MAX_VALUE, maxF3 = Double.MIN_VALUE;
        
        for (UserFeatures f : featureList) {
            minF1 = Math.min(minF1, f.totalBehaviors);
            maxF1 = Math.max(maxF1, f.totalBehaviors);
            minF2 = Math.min(minF2, f.buyCount);
            maxF2 = Math.max(maxF2, f.buyCount);
            minF3 = Math.min(minF3, f.conversionRate);
            maxF3 = Math.max(maxF3, f.conversionRate);
        }
        
        System.out.println("F1范围: [" + minF1 + ", " + maxF1 + "]");
        System.out.println("F2范围: [" + minF2 + ", " + maxF2 + "]");
        System.out.println("F3范围: [" + minF3 + ", " + maxF3 + "]");
        
        // ==================== 步骤5: 归一化 ====================
        final double f1Min = minF1, f1Max = maxF1;
        final double f2Min = minF2, f2Max = maxF2;
        final double f3Min = minF3, f3Max = maxF3;
        
        JavaRDD<String> normalizedRdd = userFeaturesRdd.map(feature -> {
            double normF1, normF2, normF3;
            
            if (f1Max == f1Min) {
                normF1 = 0.5;
            } else {
                normF1 = (feature.totalBehaviors - f1Min) / (f1Max - f1Min);
            }
            
            if (f2Max == f2Min) {
                normF2 = 0.5;
            } else {
                normF2 = (feature.buyCount - f2Min) / (f2Max - f2Min);
            }
            
            if (f3Max == f3Min) {
                normF3 = 0.5;
            } else {
                normF3 = (feature.conversionRate - f3Min) / (f3Max - f3Min);
            }
            
            return String.format("%d,%.4f,%.4f,%.4f", 
                feature.userId, normF1, normF2, normF3);
        });
        
        // ==================== 步骤6: 输出文件 ====================
        // 输出 clean_behavior_log.csv
        logsRdd.map(BehaviorLog::toString).saveAsTextFile(outputLogPath);
        System.out.println("已保存: " + outputLogPath);
        
        // 输出 user_features.csv
        normalizedRdd.saveAsTextFile(outputFeaturePath);
        System.out.println("已保存: " + outputFeaturePath);
        
        // ==================== 步骤7: 验证 ====================
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
}
